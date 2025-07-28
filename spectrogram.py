import glob
import multiprocessing as mp
import re
import argparse
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class Config:
    """
    存放所有可配置參數的預設值
    """
    # --- 輸入/輸出參數 ---
    DATA_PATH_1 = r"D:\project\ocean\output\PX"  # 第一個數據資料夾路徑
    DATA_PATH_2 = r"D:\project\ocean\output\N1"  # 第二個數據資料夾路徑（雙圖模式時使用）
    OUTPUT_DIR = r"D:\project\ocean\output\figure"  # 輸出目錄
    OUTPUT_FILENAME_BASE = 'miaoli-spectrogram_CH 20-20000'
    SAVE_FORMATS = ['pdf']
    SHOW_PLOT = False

    # --- 處理參數 ---
    START_DATE = None
    NUM_DAYS = 'all'
    DUAL_MODE = True  # False=單圖, True=雙圖

    # --- 絕對時間過濾參數 ---
    ENABLE_ABSOLUTE_TIME_FILTER = True  # 是否啟用絕對時間過濾
    DATASET_START_TIME = "2025-06-10 09:43"  # 整個數據集開始時間，格式："YYYY-MM-DD HH:MM"
    DATASET_END_TIME = "2025-07-01 21:54"  # 整個數據集結束時間，格式："YYYY-MM-DD HH:MM"


    # --- 每日時間過濾參數 ---
    ENABLE_DAILY_TIME_FILTER = False  # 是否啟用每日時間過濾
    # TIME_START = "00:00"  # 每日開始時間
    # TIME_END = "23:30"  # 每日結束時間


    # --- 圖形參數 ---
    SINGLE_FIG_SIZE = (33 / 2.54, 15 / 2.54)  # 單圖尺寸
    DUAL_FIG_SIZE = (33 / 2.54, 30 / 2.54)  # 雙圖尺寸
    FREQ_RANGE = (20, 20000)
    COLOR_RANGE = (50, 100)

    # FREQ_RANGE = None
    # COLOR_RANGE = "auto"

    # 圖表標題
    TITLE_1 = 'PX'
    TITLE_2 = 'N1'
    Y_LABEL = 'Frequency (Hz)'
    X_LABEL = 'Time (MM/DD)'
    CBAR_LABEL = r'PSD (dB re 1 $\mu$Pa$^2$ Hz$^{-1}$)'

    # --- 解析度設定 ---
    DPI = 600

    # --- 效能參數 ---
    MEMORY_LIMIT_GB = 96


def parse_date(filename):
    """從檔名解析日期"""
    match = re.search(r'\.(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', filename)
    if match:
        yy, mm, dd, hh, mi, ss = map(int, match.groups())
        return datetime(2000 + yy, mm, dd, hh, mi, ss)
    return None


def parse_absolute_time(time_str):
    """解析絕對時間字串"""
    if not time_str:
        return None
    try:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def is_daily_time_in_range(dt, start_time, end_time):
    """判斷時間是否在每日指定範圍內"""
    if not start_time or not end_time:
        return True

    try:
        time_obj = dt.time()
        start = datetime.strptime(start_time, "%H:%M").time()
        end = datetime.strptime(end_time, "%H:%M").time()

        if start <= end:  # 同一天內
            return start <= time_obj <= end
        else:  # 跨日情況
            return time_obj >= start or time_obj <= end
    except:
        return True


def get_file_info(file_path):
    """獲取檔案路徑及其對應的日期與行數"""
    date = parse_date(Path(file_path).name)
    if not date: return None
    try:
        with open(file_path, 'r') as f:
            row_count = sum(1 for row in f) - 1  # 減去標頭
        return (date, file_path, row_count)
    except Exception:
        return None


def read_data(file_path, n_cols):
    """讀取單個CSV檔案的數據"""
    try:
        return np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=range(1, n_cols + 1), dtype=np.float32)
    except Exception:
        return None


def read_batch_worker(args):
    """多核心處理的輔助函式：讀取一批檔案"""
    file_paths, n_cols = args
    return [data for path in file_paths if (data := read_data(path, n_cols)) is not None]


def downsample(data, target_points):
    """對數據進行降採樣以節省記憶體"""
    if data.shape[0] <= target_points:
        return data

    ratio = data.shape[0] / target_points
    indices = np.round(np.linspace(0, data.shape[0] - 1, target_points)).astype(int)

    downsampled = np.zeros((target_points, data.shape[1]), dtype=data.dtype)
    for i in range(target_points):
        start = int(i * ratio)
        end = min(int((i + 1) * ratio), data.shape[0])
        if start == end: end = start + 1
        downsampled[i, :] = np.mean(data[start:end, :], axis=0)

    return downsampled


def generate_ticks(start_date, end_date, n_points):
    """根據時間範圍和數據點總數，動態生成座標軸刻度與標籤"""
    positions, labels = [], []
    total_days = (end_date - start_date).days + 1

    # 根據天數決定 X 軸標籤和刻度間隔
    if total_days <= 5:
        x_label = 'Time'
        tick_interval = timedelta(hours=6)
        label_format = "%m/%d %H:%M"
        show_all_labels = True
    else:
        x_label = 'Date'
        tick_interval = timedelta(hours=12)  # 每12小時一個刻度點
        label_format = "%m/%d"
        show_all_labels = False  # 只在特定位置顯示標籤

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    total_duration_sec = (end_dt - start_dt).total_seconds()
    if total_duration_sec == 0: return [], [], 'Time'

    current_tick = start_dt
    if total_days > 5:  # 長期圖
        current_tick = datetime.combine(start_date, datetime.min.time()) + timedelta(hours=0)

    tick_count = 0
    while current_tick.date() <= end_date:
        time_since_start_sec = (current_tick - start_dt).total_seconds()
        relative_pos = time_since_start_sec / total_duration_sec
        point_index = 1 + relative_pos * (n_points - 1)

        if 1 <= point_index <= n_points:
            positions.append(point_index)

            # 對於長期圖，只在每天的第一個刻度點顯示標籤
            if show_all_labels or (tick_count % 2 == 0):
                labels.append(current_tick.strftime(label_format))
            else:
                labels.append("")  # 空標籤但保留刻度線

        current_tick += tick_interval
        tick_count += 1

    return positions, labels, x_label


def filter_files_by_absolute_time(file_info, start_time, end_time):
    """根據絕對時間過濾檔案，採用包含性策略"""
    if not start_time and not end_time:
        return file_info

    filtered_files = []
    for d, p, rc in file_info:
        # 如果設定了開始時間，檔案時間必須 >= 開始時間
        if start_time and d < start_time:
            continue
        # 如果設定了結束時間，檔案時間必須 <= 結束時間
        if end_time and d > end_time:
            continue
        filtered_files.append((d, p, rc))

    return filtered_files


def process_dataset(data_path, start_date_obj, end_date_obj, memory_limit_gb, config, dataset_name="Dataset"):
    """處理單個數據集"""
    print(f"Processing {dataset_name} from '{data_path}'...")

    # 掃描檔案
    all_files = glob.glob(str(Path(data_path) / "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in '{data_path}'")

    print(f"Found {len(all_files)} files. Parsing dates and row counts...")
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        file_info = sorted(filter(None, executor.map(get_file_info, all_files)))
    if not file_info:
        raise ValueError(f"No valid files found in '{data_path}'.")

    # 篩選日期範圍內的檔案
    selected_files_with_rows = []
    for d, p, rc in file_info:
        # 日期範圍篩選
        if not (start_date_obj <= d.date() <= end_date_obj):
            continue
        selected_files_with_rows.append((d, p, rc))

    if not selected_files_with_rows:
        raise ValueError(f"No files match the specified date range in '{data_path}'.")

    # 絕對時間過濾
    if config.ENABLE_ABSOLUTE_TIME_FILTER:
        abs_start = parse_absolute_time(config.DATASET_START_TIME)
        abs_end = parse_absolute_time(config.DATASET_END_TIME)

        if abs_start or abs_end:
            original_count = len(selected_files_with_rows)
            selected_files_with_rows = filter_files_by_absolute_time(selected_files_with_rows, abs_start, abs_end)

            if selected_files_with_rows:
                actual_start = selected_files_with_rows[0][0].strftime("%Y-%m-%d %H:%M:%S")
                actual_end = selected_files_with_rows[-1][0].strftime("%Y-%m-%d %H:%M:%S")
                print(f"Absolute time filter applied:")
                print(f"  Set range: {config.DATASET_START_TIME or 'start'} - {config.DATASET_END_TIME or 'end'}")
                print(f"  Actual range: {actual_start} - {actual_end}")
                print(f"  Files: {len(selected_files_with_rows)}/{original_count}")
            else:
                raise ValueError(f"No files match the absolute time range in '{data_path}'.")

    # 每日時間過濾
    if config.ENABLE_DAILY_TIME_FILTER:
        original_count = len(selected_files_with_rows)
        daily_filtered = []
        for d, p, rc in selected_files_with_rows:
            if is_daily_time_in_range(d, config.TIME_START, config.TIME_END):
                daily_filtered.append((d, p, rc))

        selected_files_with_rows = daily_filtered
        if selected_files_with_rows:
            print(f"Daily time filter: {config.TIME_START} - {config.TIME_END}")
            print(f"Files after daily filter: {len(selected_files_with_rows)}/{original_count}")
        else:
            raise ValueError(f"No files match the daily time range in '{data_path}'.")

    print(f"Final selection: {len(selected_files_with_rows)} files")

    # 取得頻率軸和欄位數
    with open(selected_files_with_rows[0][1], 'r') as f:
        n_cols = len(f.readline().strip().split(',')) - 1
    freq_axis = np.loadtxt(selected_files_with_rows[0][1], delimiter=',', max_rows=1, usecols=range(1, n_cols + 1),
                           dtype=np.float32)

    # 估算數據大小
    total_points = sum(rc for _, _, rc in selected_files_with_rows)
    total_size_gb = total_points * n_cols * 4 / (1024 ** 3)
    print(f"{dataset_name} size: {total_points:,} points, {total_size_gb:.2f} GB")

    # 選擇處理模式
    if total_size_gb > memory_limit_gb:
        spec_data, temp_path = process_out_of_core(selected_files_with_rows, n_cols, total_points)
    else:
        selected_files = [p for _, p, _ in selected_files_with_rows]
        spec_data = process_in_memory(selected_files, n_cols)
        temp_path = None

    return spec_data, freq_axis, temp_path


def process_in_memory(selected_files, n_cols):
    """標準的記憶體內處理流程"""
    print("Processing in-memory on CPU...")

    # 並行讀取
    batch_size = max(1, len(selected_files) // (mp.cpu_count() * 2))
    batches = [selected_files[i:i + batch_size] for i in range(0, len(selected_files), batch_size)]
    args = [(batch, n_cols) for batch in batches]
    data_list = []

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for result_batch in executor.map(read_batch_worker, args):
            data_list.extend(result_batch)

    # 堆疊數據
    spec_data = np.vstack(data_list)
    return spec_data


def process_out_of_core(selected_files_with_rows, n_cols, total_points):
    """使用記憶體映射檔案處理超大數據"""
    print("Processing out-of-core using memory-mapped file...")

    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / f"spectrogram_memmap_{int(time.time())}.mmap"
    print(f"Creating memory-mapped file at: {temp_path}")

    spec_data = np.memmap(temp_path, dtype='float32', mode='w+', shape=(total_points, n_cols))

    current_row = 0
    for i, (date, file_path, row_count) in enumerate(selected_files_with_rows):
        if i % 100 == 0:
            print(f"  -> Processing file {i + 1}/{len(selected_files_with_rows)}...")
        data = read_data(file_path, n_cols)
        if data is not None and data.shape[0] == row_count:
            spec_data[current_row: current_row + row_count, :] = data
            current_row += row_count

    spec_data.flush()
    return spec_data, temp_path


def plot_spectrogram_subplot(ax, spec_data, freq_axis, title, config, tick_positions, tick_labels):
    """繪製單個頻譜圖子圖"""
    n_points = spec_data.shape[0]
    t = np.arange(1, n_points + 1)

    if config.COLOR_RANGE == 'auto' or config.COLOR_RANGE is None:
        vmin, vmax = np.percentile(spec_data, [5, 95])
        print(f"  Auto color range for {title}: {vmin:.1f} - {vmax:.1f} dB")
    else:
        vmin, vmax = config.COLOR_RANGE
        print(f"  Manual color range for {title}: {vmin} - {vmax} dB")

    im = ax.pcolormesh(t, freq_axis, spec_data.T, cmap='jet',
                       vmin=vmin, vmax=vmax,
                       shading='auto', rasterized=True)

    # 設定軸屬性
    ax.set_yscale('log')

    # 自動或手動設定頻率範圍
    if config.FREQ_RANGE is None:
        # 自動使用數據的實際頻率範圍
        freq_min, freq_max = freq_axis.min(), freq_axis.max()
        ax.set_ylim(freq_min, freq_max)
        print(f"  Auto frequency range: {freq_min:.1f} - {freq_max:.1f} Hz")
    else:
        # 使用用戶指定的頻率範圍
        ax.set_ylim(config.FREQ_RANGE)
        print(f"  Set frequency range: {config.FREQ_RANGE[0]} - {config.FREQ_RANGE[1]} Hz")

    ax.set_xlim(1, n_points)

    # 設定刻度
    ax.tick_params(direction='out', labelsize=16, which='both')
    ax.set_axisbelow(False)

    # 設定時間刻度
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # 設定標籤
    ax.set_title(title, fontsize=18, fontweight='bold', pad=10, loc='left')
    ax.set_ylabel(config.Y_LABEL, fontsize=18)
    ax.set_xlabel(config.X_LABEL, fontsize=18)
    ax.grid(False)

    return im


def generate_spectrogram(config):
    """主函式：生成頻譜圖（支援單圖或雙圖模式）"""
    total_start_time = time.perf_counter()

    if config.DUAL_MODE:
        print("=== Generating Dual Spectrogram ===")
    else:
        print("=== Generating Single Spectrogram ===")

    # 確定日期範圍（基於第一個數據集）
    print("Determining date range from first dataset...")
    files_1 = glob.glob(str(Path(config.DATA_PATH_1) / "*.csv"))
    if not files_1:
        raise FileNotFoundError(f"No CSV files found in path '{config.DATA_PATH_1}'")

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        file_info_1 = sorted(filter(None, executor.map(get_file_info, files_1)))

    file_dates, _, _ = zip(*file_info_1)
    unique_dates = sorted(set(d.date() for d in file_dates))
    start_date_obj = unique_dates[0]

    if config.START_DATE:
        month, day = int(config.START_DATE[:2]), int(config.START_DATE[2:])
        start_date_obj = next((d for d in unique_dates if d.month == month and d.day == day), None)
        if not start_date_obj:
            raise ValueError(f"Start date {config.START_DATE} not found in data.")

    end_date_obj = unique_dates[-1]
    if config.NUM_DAYS != 'all':
        end_date_obj = start_date_obj + timedelta(days=int(config.NUM_DAYS) - 1)

    print(f"Date range: {start_date_obj} to {end_date_obj}")

    # 處理第一個數據集
    print(f"\n=== Processing First Dataset ===")
    data_1, freq_axis_1, temp_path_1 = process_dataset(
        config.DATA_PATH_1, start_date_obj, end_date_obj,
        config.MEMORY_LIMIT_GB, config, "Dataset 1"
    )

    # 生成時間刻度
    n_points = data_1.shape[0]
    tick_positions, tick_labels, x_label = generate_ticks(start_date_obj, end_date_obj, n_points)

    # 根據模式創建圖表
    print(f"\n=== Creating Plot{'s' if config.DUAL_MODE else ''} ===")
    step_start_time = time.perf_counter()

    if config.DUAL_MODE:
        # 雙圖模式
        print(f"\n=== Processing Second Dataset ===")
        data_2, freq_axis_2, temp_path_2 = process_dataset(
            config.DATA_PATH_2, start_date_obj, end_date_obj,
            config.MEMORY_LIMIT_GB, config, "Dataset 2"
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.DUAL_FIG_SIZE, sharex=True,
                                       constrained_layout=True)

        # 繪製兩個子圖
        im1 = plot_spectrogram_subplot(ax1, data_1, freq_axis_1, config.TITLE_1,
                                       config, tick_positions, tick_labels)
        im2 = plot_spectrogram_subplot(ax2, data_2, freq_axis_2, config.TITLE_2,
                                       config, tick_positions, tick_labels)

        # 為每個子圖添加colorbar
        cbar1 = fig.colorbar(im1, ax=ax1, pad=0.02, location='right', shrink=0.8)
        cbar1.ax.tick_params(labelsize=14)
        cbar1.set_label(config.CBAR_LABEL, fontsize=14)

        cbar2 = fig.colorbar(im2, ax=ax2, pad=0.02, location='right', shrink=0.8)
        cbar2.ax.tick_params(labelsize=14)
        cbar2.set_label(config.CBAR_LABEL, fontsize=14)

        temp_paths = [(temp_path_1, "Dataset 1"), (temp_path_2, "Dataset 2")]

    else:
        # 單圖模式
        fig, ax = plt.subplots(figsize=config.SINGLE_FIG_SIZE, constrained_layout=True)

        # 繪製單個圖
        im = plot_spectrogram_subplot(ax, data_1, freq_axis_1, config.TITLE_1,
                                      config, tick_positions, tick_labels)

        # 添加colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.02, location='right', shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(config.CBAR_LABEL, fontsize=14)

        temp_paths = [(temp_path_1, "Dataset 1")]

    print(f"-> Plotting finished in {time.perf_counter() - step_start_time:.2f}s")

    # 儲存檔案
    if config.SAVE_FORMATS:
        step_start_time = time.perf_counter()
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        mode_suffix = "_dual" if config.DUAL_MODE else "_single"

        # 構建時間後綴
        time_suffix = ""
        if config.ENABLE_ABSOLUTE_TIME_FILTER and (config.DATASET_START_TIME or config.DATASET_END_TIME):
            start_str = config.DATASET_START_TIME.replace("-", "").replace(":", "").replace(" ",
                                                                                            "_") if config.DATASET_START_TIME else "start"
            end_str = config.DATASET_END_TIME.replace("-", "").replace(":", "").replace(" ",
                                                                                        "_") if config.DATASET_END_TIME else "end"
            time_suffix = f"_abs_{start_str}-{end_str}"
        elif config.ENABLE_DAILY_TIME_FILTER:
            time_suffix = f"_daily_{config.TIME_START.replace(':', '')}-{config.TIME_END.replace(':', '')}"

        filename_base = f"{config.OUTPUT_FILENAME_BASE}{mode_suffix}{time_suffix}"

        print(f"Saving high-resolution output files to {output_dir.resolve()}...")
        for fmt in config.SAVE_FORMATS:
            output_path = output_dir / f"{filename_base}.{fmt}"
            plt.savefig(output_path, format=fmt, dpi=config.DPI, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"- Saved {output_path} (DPI: {config.DPI})")
        print(f"-> File saving finished in {time.perf_counter() - step_start_time:.2f}s")

    # 清理臨時檔案
    for temp_path, dataset_name in temp_paths:
        if temp_path:
            try:
                Path(temp_path).unlink()
                print(f"Temporary {dataset_name} memory-mapped file deleted")
            except Exception as e:
                print(f"Warning: Could not delete {dataset_name} temporary file. Error: {e}")

    print("\n" + "=" * 50)
    print(f"SUCCESS! Total processing time: {time.perf_counter() - total_start_time:.2f}s")
    if config.DUAL_MODE:
        print(f"Processed Dataset 1: {data_1.shape[0]:,} points, Dataset 2: {data_2.shape[0]:,} points")
    else:
        print(f"Processed Dataset: {data_1.shape[0]:,} points")
    print(f"Output resolution: {config.DPI} DPI")

    if config.SHOW_PLOT:
        print("Displaying plot...")
        plt.show()

    print("=" * 50)


def main():
    """主程式進入點，負責解析命令列參數並執行"""
    config = Config()

    parser = argparse.ArgumentParser(description="Generate spectrogram from CSV data (single or dual mode).")
    parser.add_argument('-d1', '--data_path_1', type=str,
                        help=f"Path to first data directory. Default: '{config.DATA_PATH_1}'")
    parser.add_argument('-d2', '--data_path_2', type=str,
                        help=f"Path to second data directory (dual mode only). Default: '{config.DATA_PATH_2}'")
    parser.add_argument('-o', '--output_dir', type=str,
                        help=f"Directory to save output files. Default: '{config.OUTPUT_DIR}'")
    parser.add_argument('--dual', action='store_true',
                        help="Enable dual mode (plot two datasets)")
    parser.add_argument('--title1', type=str, help=f"Title for first plot. Default: '{config.TITLE_1}'")
    parser.add_argument('--title2', type=str, help=f"Title for second plot. Default: '{config.TITLE_2}'")
    parser.add_argument('--freq-range', type=str,
                        help="Frequency range as 'min,max' (e.g., '20,20000'). Default: auto.")
    parser.add_argument('--start', type=str, help="Start date for processing in 'MMDD' format. Default: auto.")
    parser.add_argument('--days', type=str, help="Number of days to process. Default: 'all'.")

    # 絕對時間過濾參數
    parser.add_argument('--abs-start', type=str,
                        help="Absolute start time 'YYYY-MM-DD HH:MM' (e.g., '2025-07-24 09:43')")
    parser.add_argument('--abs-end', type=str,
                        help="Absolute end time 'YYYY-MM-DD HH:MM' (e.g., '2025-07-24 20:57')")
    parser.add_argument('--abs-time-range', type=str,
                        help="Absolute time range 'start,end' (e.g., '2025-07-24 09:43,2025-07-24 20:57')")

    # 每日時間過濾參數
    parser.add_argument('--daily-time-range', type=str,
                        help="Daily time range as 'HH:MM,HH:MM' (e.g., '10:00,21:00')")
    parser.add_argument('--enable-daily-filter', action='store_true',
                        help="Enable daily time filtering with default range (10:00-21:00)")

    parser.add_argument('--dpi', type=int, help=f"Output DPI for high resolution. Default: {config.DPI}")
    parser.add_argument('--no-save', action='store_true', help="Disable saving output files.")
    parser.add_argument('--show', action='store_true', help="Show the plot window after processing.")

    args = parser.parse_args()

    if args.data_path_1: config.DATA_PATH_1 = args.data_path_1
    if args.data_path_2: config.DATA_PATH_2 = args.data_path_2
    if args.output_dir: config.OUTPUT_DIR = args.output_dir
    if args.dual: config.DUAL_MODE = True
    if args.title1: config.TITLE_1 = args.title1
    if args.title2: config.TITLE_2 = args.title2
    if args.freq_range:
        try:
            freq_min, freq_max = map(float, args.freq_range.split(','))
            config.FREQ_RANGE = (freq_min, freq_max)
        except ValueError:
            print("Warning: Invalid frequency range format. Using auto range.")
    if args.start: config.START_DATE = args.start
    if args.days: config.NUM_DAYS = args.days

    # 處理絕對時間過濾參數
    if args.abs_start:
        config.DATASET_START_TIME = args.abs_start
        config.ENABLE_ABSOLUTE_TIME_FILTER = True
    if args.abs_end:
        config.DATASET_END_TIME = args.abs_end
        config.ENABLE_ABSOLUTE_TIME_FILTER = True
    if args.abs_time_range:
        try:
            start_time, end_time = args.abs_time_range.split(',')
            config.DATASET_START_TIME = start_time.strip()
            config.DATASET_END_TIME = end_time.strip()
            config.ENABLE_ABSOLUTE_TIME_FILTER = True
        except ValueError:
            print("Warning: Invalid absolute time range format.")

    # 處理每日時間過濾參數
    if args.daily_time_range:
        try:
            time_start, time_end = args.daily_time_range.split(',')
            config.TIME_START = time_start.strip()
            config.TIME_END = time_end.strip()
            config.ENABLE_DAILY_TIME_FILTER = True
        except ValueError:
            print("Warning: Invalid daily time range format.")
    if args.enable_daily_filter:
        config.ENABLE_DAILY_TIME_FILTER = True

    if args.dpi: config.DPI = args.dpi
    if args.no_save: config.SAVE_FORMATS = []
    if args.show: config.SHOW_PLOT = True

    try:
        generate_spectrogram(config)
    except Exception as e:
        print(f"\nError occurred: {e}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()