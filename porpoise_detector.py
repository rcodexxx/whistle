#!/usr/bin/env python3
"""
Porpoise Click Detection System - Producer-Consumer Pattern
"""

import numpy as np
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
import argparse
import pandas as pd
import glob
from tqdm import tqdm
import multiprocessing as mp
import threading
import queue
import time


def cpu_worker(read_queue, write_queue, analyzer_params, worker_id):
    """CPU工作進程"""
    analyzer = PorpoiseClickAnalyzer()
    analyzer.__dict__.update(analyzer_params)

    while True:
        try:
            item = read_queue.get(timeout=30)
            if item is None:  # 結束信號
                break

            file_path, rel_path, data, sample_rate = item

            if data is None:  # 讀取失敗
                write_queue.put((file_path, rel_path, None))
                continue

            # 處理數據
            result = analyzer.analyze_data(file_path, data, sample_rate)
            if result:
                result['relative_path'] = rel_path
            write_queue.put((file_path, rel_path, result))

            # 釋放記憶體
            del data

        except queue.Empty:
            # timeout是正常的，繼續等待
            continue
        except Exception as e:
            # 處理錯誤但不退出worker
            print(f"Worker {worker_id} processing error: {e}")
            try:
                write_queue.put((file_path, rel_path, None))
            except:
                pass
            continue


def io_reader(files_with_rel_paths, read_queue):
    """I/O讀取線程"""
    for file_path, rel_path in files_with_rel_paths:
        try:
            data, sample_rate = sf.read(file_path, dtype=np.float32)
            read_queue.put((file_path, rel_path, data, sample_rate))
        except Exception:
            read_queue.put((file_path, rel_path, None, None))


def io_writer(write_queue, output_dir, analyzer, total_files, progress_bar):
    """I/O寫入線程"""
    processed = 0
    detections = 0
    files_with_detections = 0

    while processed < total_files:
        try:
            file_path, rel_path, result = write_queue.get(timeout=60)

            if result:
                analyzer.save_file_results(result, output_dir)
                if result['valid_groups']:
                    detections += len(result['valid_groups'])
                    files_with_detections += 1

            processed += 1
            progress_bar.update(1)

        except queue.Empty:
            continue

    progress_bar.close()
    print(f"Completed: {files_with_detections}/{total_files} files with detections, {detections} total detections")


class PorpoiseClickAnalyzer:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers

        # 信號檢測參數
        self.threshold_factor = 3.5  # 動態閾值倍數 (信號/背景噪音比例)

        # 單個點擊聲參數 (根據江豚NBHF特徵)
        self.min_duration = 0.00005  # 最短持續時間: 50微秒
        self.max_duration = 0.000175  # 最長持續時間: 175微秒
        self.min_click_interval = 0.001  # 相鄰點擊聲最小間隔: 1毫秒 (避免重複檢測)

        # 群組化參數 (回聲定位點擊串)
        self.min_clicks_per_group = 6  # 群組最少點擊聲數量
        self.max_clicks_per_group = 100  # 群組最多點擊聲數量 (避免長時間噪音)
        self.max_group_interval = 0.03  # 群組化最大間隔: 30毫秒
        self.min_group_duration = 0.005  # 群組最短時間跨度: 5毫秒
        self.max_group_duration = 0.2  # 群組最長時間跨度: 200毫秒

        # 品質過濾參數
        self.min_snr_db = 20  # 最低信噪比要求: 20分貝

    def analyze_data(self, file_path, data, sample_rate):
        """分析音頻數據"""
        try:
            if data.ndim != 2 or data.shape[1] != 2:
                return None

            iq_data = data[:, 0] + 1j * data[:, 1]
            envelope = np.abs(iq_data)
            background_level = np.percentile(envelope, 5)
            threshold_level = background_level * self.threshold_factor

            candidates = self._find_candidates(envelope, threshold_level, sample_rate)
            validated_clicks, validation_stats = self._validate_clicks(candidates, iq_data, sample_rate,
                                                                       background_level)
            valid_groups = self._group_and_filter(validated_clicks)

            return {
                'filename': Path(file_path).name,
                'valid_groups': valid_groups,
                'background_level': background_level,
                'threshold_level': threshold_level,
                'signal_max': np.max(envelope),
                'signal_min': np.min(envelope),
                'signal_mean': np.mean(envelope),
                'stats': validation_stats
            }
        except Exception:
            return None

    def _find_candidates(self, envelope, threshold_level, sample_rate):
        """尋找候選點擊聲"""
        above_threshold = envelope > threshold_level
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(above_threshold)]])

        durations = (ends - starts) / sample_rate
        valid_mask = (durations >= self.min_duration) & (durations <= self.max_duration)

        candidates = []
        for i in range(len(starts)):
            if valid_mask[i]:
                candidates.append({
                    'start': starts[i],
                    'end': ends[i],
                    'duration': durations[i],
                    'start_time': starts[i] / sample_rate,
                    'max_amplitude': np.max(envelope[starts[i]:ends[i]])
                })
        return candidates

    def _validate_clicks(self, candidates, iq_data, sample_rate, background_level):
        """驗證點擊聲品質"""
        validated = []
        last_time = -float('inf')
        context_samples = int(sample_rate * 0.0005)

        stats = {'total_candidates': len(candidates), 'validated': 0, 'snr_values': []}

        for click in candidates:
            if click['start_time'] - last_time < self.min_click_interval:
                continue

            signal_amplitude = click['max_amplitude']
            snr_db = 20 * np.log10(signal_amplitude / background_level) if background_level > 0 else 0

            stats['snr_values'].append(snr_db)

            if snr_db < self.min_snr_db:
                continue

            start_idx = max(0, click['start'] - context_samples)
            end_idx = min(len(iq_data), click['end'] + context_samples)
            segment = iq_data[start_idx:end_idx]

            if len(segment) < 32:
                continue

            try:
                nperseg = min(128, len(segment))
                freqs, psd = signal.welch(segment, sample_rate, nperseg=nperseg)
                peak_idx = np.argmax(psd)
                peak_freq = abs(freqs[peak_idx])

                half_max = psd[peak_idx] / 2
                above_half = psd > half_max
                if np.sum(above_half) > 1:
                    freq_indices = np.where(above_half)[0]
                    bandwidth = freqs[freq_indices[-1]] - freqs[freq_indices[0]]
                else:
                    bandwidth = 0

                click.update({
                    'peak_frequency': peak_freq,
                    'bandwidth': bandwidth,
                    'snr_db': snr_db
                })
                validated.append(click)
                stats['validated'] += 1
                last_time = click['start_time']

            except Exception:
                click.update({'peak_frequency': 0, 'bandwidth': 0, 'snr_db': snr_db})
                validated.append(click)
                stats['validated'] += 1
                last_time = click['start_time']

        return validated, stats

    def _group_and_filter(self, validated_clicks):
        """群組化點擊聲"""
        if len(validated_clicks) < self.min_clicks_per_group:
            return []

        clicks_sorted = sorted(validated_clicks, key=lambda x: x['start_time'])
        groups = []
        current_group = [clicks_sorted[0]]

        for i in range(1, len(clicks_sorted)):
            interval = clicks_sorted[i]['start_time'] - current_group[-1]['start_time']
            if interval <= self.max_group_interval:
                current_group.append(clicks_sorted[i])
            else:
                valid_group = self._create_group(current_group)
                if valid_group:
                    groups.append(valid_group)
                current_group = [clicks_sorted[i]]

        valid_group = self._create_group(current_group)
        if valid_group:
            groups.append(valid_group)
        return groups

    def _create_group(self, clicks):
        """建立群組資料結構"""
        if len(clicks) < self.min_clicks_per_group:
            return None

        duration = clicks[-1]['start_time'] - clicks[0]['start_time']
        if not (self.min_group_duration <= duration <= self.max_group_duration):
            return None
        if len(clicks) > self.max_clicks_per_group:
            return None

        snr_values = [c.get('snr_db', 0) for c in clicks]
        freq_values = [c.get('peak_frequency', 0) for c in clicks if c.get('peak_frequency', 0) > 0]
        bandwidth_values = [c.get('bandwidth', 0) for c in clicks if c.get('bandwidth', 0) > 0]

        return {
            'start_time': clicks[0]['start_time'],
            'end_time': clicks[-1]['start_time'],
            'duration': duration,
            'num_clicks': len(clicks),
            'mean_snr_db': np.mean(snr_values) if snr_values else 0,
            'mean_frequency_hz': np.mean(freq_values) if freq_values else 0,
            'mean_bandwidth_hz': np.mean(bandwidth_values) if bandwidth_values else 0
        }

    def save_file_results(self, result, output_dir):
        """儲存結果"""
        rel_path = result.get('relative_path', '')
        filename = Path(result['filename'])
        filename_stem = filename.stem

        output_dir = Path(output_dir)

        # 建立對應的子目錄結構
        if rel_path:
            rel_dir = Path(rel_path).parent
            detection_dir = output_dir / "detections" / rel_dir
            stats_dir = output_dir / "stats" / rel_dir
        else:
            detection_dir = output_dir / "detections"
            stats_dir = output_dir / "stats"

        detection_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)

        # 儲存偵測結果
        detections_file = detection_dir / f"{filename_stem}.csv"
        csv_data = []
        if result['valid_groups']:
            for i, group in enumerate(result['valid_groups']):
                csv_data.append({
                    'group_id': i + 1,
                    'start_time': group['start_time'],
                    'end_time': group['end_time'],
                    'duration_ms': group['duration'] * 1000,
                    'num_clicks': group['num_clicks'],
                    'mean_snr_db': group['mean_snr_db'],
                    'mean_frequency_hz': group['mean_frequency_hz'],
                    'mean_bandwidth_hz': group['mean_bandwidth_hz']
                })
        else:
            csv_data.append({
                'group_id': 0, 'start_time': 0, 'end_time': 0, 'duration_ms': 0,
                'num_clicks': 0, 'mean_snr_db': 0,
                'mean_frequency_hz': 0, 'mean_bandwidth_hz': 0
            })

        pd.DataFrame(csv_data).to_csv(detections_file, index=False)

        # 儲存統計
        stats_file = stats_dir / f"{filename_stem}.csv"
        stats = result['stats']
        stats_data = [{
            'filename': result['filename'],
            'relative_path': rel_path,
            'background_level': result['background_level'],
            'threshold_level': result['threshold_level'],
            'signal_max': result['signal_max'],
            'signal_min': result['signal_min'],
            'signal_mean': result['signal_mean'],
            'total_candidates': stats['total_candidates'],
            'validated_clicks': stats['validated'],
            'final_groups': len(result['valid_groups']),
            'snr_mean': np.mean(stats['snr_values']) if stats['snr_values'] else 0,
            'snr_min': np.min(stats['snr_values']) if stats['snr_values'] else 0,
            'snr_max': np.max(stats['snr_values']) if stats['snr_values'] else 0
        }]
        pd.DataFrame(stats_data).to_csv(stats_file, index=False)

    def batch_process(self, input_path, output_dir):
        """生產者-消費者模式批次處理"""
        input_path = Path(input_path)

        if input_path.is_dir():
            files = sorted(input_path.glob("**/*.wav"))  # 遞歸搜尋所有子目錄
            # 計算相對路徑
            files_with_rel_paths = []
            for file_path in files:
                rel_path = file_path.relative_to(input_path)
                files_with_rel_paths.append((file_path, str(rel_path)))
        else:
            files = sorted(glob.glob(str(input_path)))
            # 對於檔案模式，不保留目錄結構
            files_with_rel_paths = [(Path(f), Path(f).name) for f in files]

        if not files_with_rel_paths:
            print("No WAV files found")
            return

        print(f"Processing {len(files_with_rel_paths)} files")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 建立佇列
        read_queue = mp.Queue(maxsize=32)  # 讀取佇列，增加容量
        write_queue = mp.Queue()  # 寫入佇列

        # 準備分析器參數
        analyzer_params = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k != 'num_workers'}

        # 啟動CPU工作進程
        cpu_processes = []
        for i in range(self.num_workers):
            p = mp.Process(target=cpu_worker, args=(read_queue, write_queue, analyzer_params, i))
            p.start()
            cpu_processes.append(p)

        # 建立進度條
        progress_bar = tqdm(total=len(files_with_rel_paths), desc="Processing")

        # 啟動I/O線程
        reader_thread = threading.Thread(target=io_reader, args=(files_with_rel_paths, read_queue))
        writer_thread = threading.Thread(target=io_writer,
                                         args=(write_queue, output_dir, self, len(files_with_rel_paths), progress_bar))

        reader_thread.start()
        writer_thread.start()

        # 等待讀取完成
        reader_thread.join()

        # 發送結束信號給CPU進程
        for _ in range(self.num_workers):
            read_queue.put(None)

        # 等待CPU進程完成
        for p in cpu_processes:
            p.join()

        # 等待寫入完成
        writer_thread.join()


def main():
    parser = argparse.ArgumentParser(description='Porpoise Click Detection System - Producer-Consumer')
    parser.add_argument('input', help='Input directory or file pattern')
    parser.add_argument('--output-dir', '-o', default='./results', help='Output directory')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of CPU worker processes')
    args = parser.parse_args()

    analyzer = PorpoiseClickAnalyzer(num_workers=args.workers)
    analyzer.batch_process(args.input, args.output_dir)


if __name__ == '__main__':
    main()