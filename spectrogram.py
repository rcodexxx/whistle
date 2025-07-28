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

# --- 可選的 GPU 加速 ---
try:
    import cupy as cp

    GPU_ENABLED = True
except ImportError:
    cp = None
    GPU_ENABLED = False


class Config:
    """
    存放所有可配置參數的預設值
    """
    # --- 輸入/輸出參數 ---
    DATA_PATH = './'
    OUTPUT_DIR = './'  # 輸出目錄
    # 修正：使用固定的檔名和標題，與 MATLAB 腳本匹配
    OUTPUT_FILENAME_BASE = 'long-term_spectrogram CH 20-20000'
    SAVE_FORMATS = ['pdf', 'png']
    SHOW_PLOT = False

    # --- 處理參數 ---
    START_DATE = None
    NUM_DAYS = 'all'

    # --- 圖形參數 (與 MATLAB 腳本對齊) ---
    # 修正：精確匹配 MATLAB 的 33cm x 15cm 紙張尺寸
    FIG_SIZE = (33 / 2.54, 15 / 2.54)  # (寬, 高) in inches
    FREQ_RANGE = (60, 20000)
    COLOR_RANGE = (50, 100)
    # 視覺一致性精修
    TITLE = 'long-term_spectrogram CH 20-20000'  # 圖上固定的標題
    Y_LABEL = 'Frequency (Hz)'
    X_LABEL = 'Time (MM/DD)'  # 預設標籤，會由 generate_ticks 動態覆寫
    # 使用 LaTeX 格式精確呈現單位
    CBAR_LABEL = r'PSD (dB re 1 $\mu$Pa$^2$ Hz$^{-1}$)'

    # --- 效能參數 ---
    USE_GPU = True
    MEMORY_LIMIT_GB = 96


def parse_date(filename):
    """從檔名解析日期"""
    match = re.search(r'\.(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', filename)
    if match:
        yy, mm, dd, hh, mi, ss = map(int, match.groups())
        return datetime(2000 + yy, mm, dd, hh, mi, ss)
    return None


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


def downsample(data, target_points, xp=np):
    """對數據進行降採樣以節省記憶體，支援 CPU/GPU"""
    if data.shape[0] <= target_points:
        return data

    ratio = data.shape[0] / target_points
    indices = xp.round(xp.linspace(0, data.shape[0] - 1, target_points)).astype(int)

    downsampled = xp.zeros((target_points, data.shape[1]), dtype=data.dtype)
    for i in range(target_points):
        start = int(i * ratio)
        end = min(int((i + 1) * ratio), data.shape[0])
        if start == end: end = start + 1
        downsampled[i, :] = xp.mean(data[start:end, :], axis=0)

    return downsampled


def generate_ticks(start_date, end_date, n_points):
    """根據時間範圍和數據點總數，動態生成座標軸刻度與標籤"""
    positions, labels = [], []
    total_days = (end_date - start_date).days + 1

    # 根據天數決定 X 軸標籤和刻度間隔
    if total_days <= 5:
        x_label = 'Time'
        interval = timedelta(hours=6)
        label_format = "%m/%d %H:%M"
    else:
        x_label = 'Date'
        interval = timedelta(days=1)
        label_format = "%m/%d"

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    total_duration_sec = (end_dt - start_dt).total_seconds()
    if total_duration_sec == 0: return [], [], 'Time'

    current_tick = start_dt
    if total_days > 5:  # 長期圖，標籤置於中午，視覺上更置中
        current_tick = datetime.combine(start_date, datetime.min.time()) + timedelta(hours=12)

    while current_tick.date() <= end_date:
        time_since_start_sec = (current_tick - start_dt).total_seconds()
        relative_pos = time_since_start_sec / total_duration_sec
        point_index = 1 + relative_pos * (n_points - 1)

        if 1 <= point_index <= n_points:
            positions.append(point_index)
            labels.append(current_tick.strftime(label_format))
        current_tick += interval

    return positions, labels, x_label


def process_in_memory(selected_files, n_cols, use_gpu):
    """標準的記憶體內處理流程"""
    xp = cp if use_gpu and GPU_ENABLED else np
    device = "GPU" if xp == cp else "CPU"
    print(f"Processing in-memory on {device}...")

    # 並行讀取
    batch_size = max(1, len(selected_files) // (mp.cpu_count() * 2))
    batches = [selected_files[i:i + batch_size] for i in range(0, len(selected_files), batch_size)]
    args = [(batch, n_cols) for batch in batches]
    data_list = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for result_batch in executor.map(read_batch_worker, args):
            data_list.extend(result_batch)

    # 堆疊
    if xp == cp:
        data_list_gpu = [cp.array(d) for d in data_list]
        spec_data = xp.vstack(data_list_gpu)
    else:
        spec_data = np.vstack(data_list)

    return spec_data, xp


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
    return spec_data, np, temp_path


def generate_spectrogram(config):
    """主函式：生成頻譜圖"""
    total_start_time = time.perf_counter()

    # 1. 掃描並解析檔案
    step_start_time = time.perf_counter()
    print(f"Scanning for *.csv files in '{config.DATA_PATH}'...")
    all_files = glob.glob(str(Path(config.DATA_PATH) / "*.csv"))
    if not all_files: raise FileNotFoundError(f"No CSV files found in '{config.DATA_PATH}'")

    print(f"Found {len(all_files)} files. Parsing dates and row counts...")
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        file_info = sorted(filter(None, executor.map(get_file_info, all_files)))
    if not file_info: raise ValueError("No valid files found.")
    print(f"-> File pre-scan finished in {time.perf_counter() - step_start_time:.2f}s")

    # 2. 確定日期範圍並篩選檔案
    file_dates, _, _ = zip(*file_info)
    unique_dates = sorted(set(d.date() for d in file_dates))
    start_date_obj = unique_dates[0]
    if config.START_DATE:
        month, day = int(config.START_DATE[:2]), int(config.START_DATE[2:])
        start_date_obj = next((d for d in unique_dates if d.month == month and d.day == day), None)
        if not start_date_obj: raise ValueError(f"Start date {config.START_DATE} not found in data.")

    end_date_obj = unique_dates[-1]
    if config.NUM_DAYS != 'all':
        end_date_obj = start_date_obj + timedelta(days=int(config.NUM_DAYS) - 1)

    selected_files_with_rows = [(d, p, rc) for d, p, rc in file_info if start_date_obj <= d.date() <= end_date_obj]
    if not selected_files_with_rows: raise ValueError("No files match the specified date range.")

    # 3. 估算總大小並選擇處理模式
    with open(selected_files_with_rows[0][1], 'r') as f:
        n_cols = len(f.readline().strip().split(',')) - 1
    freq_axis = np.loadtxt(selected_files_with_rows[0][1], delimiter=',', max_rows=1, usecols=range(1, n_cols + 1),
                           dtype=np.float32)

    total_points = sum(rc for _, _, rc in selected_files_with_rows)
    total_size_gb = total_points * n_cols * 4 / (1024 ** 3)
    print(f"Total data points: {total_points:,}. Estimated size: {total_size_gb:.2f} GB")

    spec_data = None
    xp = np
    temp_mmap_path = None

    if total_size_gb > config.MEMORY_LIMIT_GB:
        spec_data, xp, temp_mmap_path = process_out_of_core(selected_files_with_rows, n_cols, total_points)
    else:
        selected_files = [p for _, p, _ in selected_files_with_rows]
        spec_data, xp = process_in_memory(selected_files, n_cols, config.USE_GPU)

    # 4. 繪製頻譜圖
    step_start_time = time.perf_counter()
    print("Plotting spectrogram...")
    n_points = spec_data.shape[0]
    fig, ax = plt.subplots(figsize=config.FIG_SIZE)

    t = np.arange(1, n_points + 1)

    plot_data = spec_data.get() if xp == cp else spec_data
    im = ax.pcolormesh(t, freq_axis, plot_data.T, cmap='jet', vmin=config.COLOR_RANGE[0], vmax=config.COLOR_RANGE[1],
                       shading='auto', rasterized=True)

    im.set_rasterized(True)
    im.set_zorder(0)

    ax.set_yscale('log')
    ax.set_ylim(config.FREQ_RANGE)
    ax.set_xlim(1, n_points)
    ax.tick_params(labelsize=18)  # 匹配 MATLAB fontsize

    tick_positions, tick_labels, x_label = generate_ticks(start_date_obj, end_date_obj, n_points)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # 使用 Config 中的固定標題
    if config.TITLE:
        ax.set_title(config.TITLE, fontsize=16)

    ax.set_ylabel(config.Y_LABEL, fontsize=20)
    ax.set_xlabel(x_label, fontsize=20)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(config.CBAR_LABEL, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    print(f"-> Plotting finished in {time.perf_counter() - step_start_time:.2f}s")

    # 5. 儲存檔案
    if config.SAVE_FORMATS:
        step_start_time = time.perf_counter()

        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 使用 Config 中的固定檔名
        filename_base = config.OUTPUT_FILENAME_BASE

        print(f"Saving output files to {output_dir.resolve()}...")
        for fmt in config.SAVE_FORMATS:
            output_path = output_dir / f"{filename_base}.{fmt}"
            plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
            print(f"- Saved {output_path}")
        print(f"-> File saving finished in {time.perf_counter() - step_start_time:.2f}s")
    else:
        print("File saving skipped by configuration.")

    # 6. 清理
    if temp_mmap_path:
        del spec_data
        try:
            Path(temp_mmap_path).unlink()
            print(f"Temporary memory-mapped file deleted: {temp_mmap_path}")
        except Exception as e:
            print(f"Warning: Could not delete temporary file {temp_mmap_path}. Error: {e}")

    print("-" * 30)
    print(f"Success! Total time: {time.perf_counter() - total_start_time:.2f}s")
    print(f"Processed {n_points:,} time points.")

    if config.SHOW_PLOT:
        print("Displaying plot...")
        plt.show()

    print("-" * 30)


def main():
    """主程式進入點，負責解析命令列參數並執行"""
    config = Config()

    parser = argparse.ArgumentParser(description="Generate a long-term spectrogram from CSV data.")
    parser.add_argument('-d', '--data_path', type=str,
                        help=f"Path to the directory containing CSV files. Default: '{config.DATA_PATH}'")
    parser.add_argument('-o', '--output_dir', type=str,
                        help=f"Directory to save output files. Default: '{config.OUTPUT_DIR}'")
    parser.add_argument('--start', type=str, help="Start date for processing in 'MMDD' format. Default: auto.")
    parser.add_argument('--days', type=str, help="Number of days to process. Default: 'all'.")
    parser.add_argument('--no-gpu', action='store_true', help="Disable GPU acceleration and force CPU usage.")
    parser.add_argument('--no-save', action='store_true', help="Disable saving output files.")
    parser.add_argument('--show', action='store_true', help="Show the plot window after processing.")

    args = parser.parse_args()

    if args.data_path: config.DATA_PATH = args.data_path
    if args.output_dir: config.OUTPUT_DIR = args.output_dir
    if args.start: config.START_DATE = args.start
    if args.days: config.NUM_DAYS = args.days
    if args.no_gpu: config.USE_GPU = False
    if args.no_save: config.SAVE_FORMATS = []
    if args.show: config.SHOW_PLOT = True

    try:
        generate_spectrogram(config)
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
