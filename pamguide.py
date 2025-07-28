#!/usr/bin/env python3

import argparse
import time
import queue
import threading
from pathlib import Path

from dataclasses import dataclass
from typing import Tuple, Optional
import os
import gc

import numpy as np
import soundfile as sf
import scipy as sp


@dataclass(frozen=True)
class Config:
    sensitivity: float = -176.3
    window_sec: float = 0.001
    overlap: float = 0.5
    freq_range: Tuple[float, float] = (110000.0, 192000.0)
    time_avg: float = 0


@dataclass
class Task:
    path: Path
    signal: np.ndarray
    fs: float


class OptimizedProcessor:
    """記憶體優化處理器 - 預分配buffer重用"""

    def __init__(self, config: Config):
        self.config = config
        self._pref = 1.0
        self.min_power = 1e-30
        self.max_db = 300.0
        self.min_db = -300.0

        # 預分配buffer
        max_samples = 30 * 60 * 192000
        self.signal_buffer = np.zeros(max_samples, dtype=np.float32)

    def process(self, signal: np.ndarray, fs: float) -> np.ndarray:
        # 重用signal buffer
        signal_len = len(signal)
        if signal_len <= len(self.signal_buffer):
            self.signal_buffer[:signal_len] = signal
            working_signal = self.signal_buffer[:signal_len]
        else:
            # fallback到原始方式
            working_signal = signal

        working_signal = self._clean_signal_inplace(working_signal)

        window_samples = int(self.config.window_sec * fs)
        overlap_samples = int(np.ceil(window_samples * self.config.overlap))

        if window_samples > len(working_signal):
            window_samples = len(working_signal)
            overlap_samples = int(window_samples * self.config.overlap)

        # 核心頻譜計算
        freqs, times, sxx = sp.signal.spectrogram(
            working_signal,
            fs=fs,
            window='hann',
            nperseg=window_samples,
            noverlap=overlap_samples,
            scaling='density',
            mode='psd'
        )

        # 立即清理大陣列
        del working_signal
        del signal

        sxx = self._clean_spectrogram(sxx)

        # 頻率過濾
        freq_mask = (freqs >= self.config.freq_range[0]) & (freqs <= self.config.freq_range[1])
        filtered_freqs = freqs[freq_mask]
        filtered_sxx = sxx[freq_mask, :]

        # 清理原始數據
        del freqs, sxx

        if filtered_sxx.size == 0:
            return np.zeros((2, 2), dtype=np.float32)

        # dB轉換
        psd_db = self._safe_db_conversion(filtered_sxx.T) - self.config.sensitivity

        # 清理中間結果
        del filtered_sxx

        # 時間平均
        final_psd = psd_db
        final_times = times
        if self.config.time_avg > 0 and len(times) > 1:
            step = times[1] - times[0]
            if step > 0:
                final_psd, final_times = self._welch_avg(psd_db, times, step)

        return self._build_matrix(final_psd, filtered_freqs, final_times)

    def _clean_signal_inplace(self, signal: np.ndarray) -> np.ndarray:
        """就地清理信號，不分配新記憶體"""
        if signal is None or signal.size == 0:
            raise ValueError("信號為空")

        # 就地操作
        np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.clip(signal, -10.0, 10.0, out=signal)

        return signal

    def _clean_spectrogram(self, sxx: np.ndarray) -> np.ndarray:
        sxx = np.nan_to_num(sxx, nan=self.min_power, posinf=1e10, neginf=self.min_power)
        sxx = np.maximum(sxx, self.min_power)
        return sxx.astype(np.float32)

    def _safe_db_conversion(self, filtered_sxx: np.ndarray) -> np.ndarray:
        safe_sxx = np.maximum(filtered_sxx, self.min_power)
        psd_db = 10 * np.log10(safe_sxx / (self._pref ** 2))
        psd_db = np.clip(psd_db, self.min_db, self.max_db)
        psd_db = np.nan_to_num(psd_db, nan=self.min_db, posinf=self.max_db, neginf=self.min_db)
        return psd_db

    def _welch_avg(self, psd_db: np.ndarray, original_times: np.ndarray, step: float):
        if step == 0:
            return psd_db, original_times

        factor = int(round(self.config.time_avg / step))
        if factor <= 1:
            return psd_db, original_times

        psd_db = np.nan_to_num(psd_db, nan=self.min_db, posinf=self.max_db, neginf=self.min_db)
        psd_db = np.clip(psd_db, self.min_db, self.max_db)

        linear_power = 10 ** (psd_db / 10)
        linear_power = np.clip(linear_power, self.min_power, 1e10)

        n_rows, n_cols = linear_power.shape
        n_out = int(np.ceil(n_rows / factor))

        target_rows = n_out * factor
        padding_needed = target_rows - n_rows
        if padding_needed > 0:
            padding = np.full((padding_needed, n_cols), self.min_power, dtype=np.float32)
            linear_power = np.vstack([linear_power, padding])

        reshaped_power = linear_power.reshape(n_out, factor, n_cols)
        avg_power = np.mean(reshaped_power, axis=1)

        avg_power = np.clip(avg_power, self.min_power, 1e10)
        avg_power = np.nan_to_num(avg_power, nan=self.min_power, posinf=1e10, neginf=self.min_power)

        avg_psd_db = 10 * np.log10(avg_power)
        avg_psd_db = np.clip(avg_psd_db, self.min_db, self.max_db)
        avg_psd_db = np.nan_to_num(avg_psd_db, nan=self.min_db, posinf=self.max_db, neginf=self.min_db)

        new_times = original_times[::factor][:n_out]
        new_times = np.nan_to_num(new_times, nan=0.0, posinf=3600.0, neginf=0.0)

        return avg_psd_db.astype(np.float32), new_times.astype(np.float32)

    def _build_matrix(self, psd: np.ndarray, freqs: np.ndarray, times: np.ndarray) -> np.ndarray:
        psd = np.nan_to_num(psd, nan=self.min_db, posinf=self.max_db, neginf=self.min_db)
        psd = np.clip(psd, self.min_db, self.max_db)

        freqs = np.nan_to_num(freqs, nan=0.0, posinf=self.config.freq_range[1], neginf=0.0)
        freqs = np.clip(freqs, 0.0, self.config.freq_range[1])

        times = np.nan_to_num(times, nan=0.0, posinf=3600.0, neginf=0.0)
        times = np.clip(times, 0.0, 3600.0)

        rows, cols = psd.shape
        matrix = np.zeros((rows + 1, cols + 1), dtype=np.float32)
        matrix[0, 0] = 2211
        matrix[0, 1:] = freqs
        matrix[1:, 0] = times
        matrix[1:, 1:] = psd

        matrix = np.nan_to_num(matrix, nan=0.0, posinf=self.max_db, neginf=self.min_db)
        return matrix


class Runner:
    def __init__(self, config: Config, output_dir: Optional[Path] = None):
        self.config = config
        self.output_dir = output_dir
        self.processor = OptimizedProcessor(config)  # 使用優化處理器

        # 經過驗證的穩定配置
        self.num_workers = 5
        self.load_q = queue.Queue(maxsize=7)  # 6+2
        self.write_q = queue.Queue(maxsize=14)  # 6*2

        self.done = 0
        self.fail = 0
        self.active = 0
        self.total = 0
        self.start = 0
        self.last_update = 0
        self.lock = threading.Lock()

        self.stop_flag = threading.Event()
        self.reader_finished = threading.Event()

    def run(self, paths):
        self.total = len(paths)
        self.start = time.time()

        print(f"文件: {self.total}")
        print(f"Workers: {self.num_workers}")
        print()

        self._update()

        # 標準三線程架構
        threads = [
            threading.Thread(target=self.reader, args=(paths,), name="Reader"),
            *[threading.Thread(target=self.worker, name=f"Worker-{i + 1}")
              for i in range(self.num_workers)],
            threading.Thread(target=self.writer, name="Writer")
        ]

        for t in threads:
            t.daemon = True
            t.start()

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            self.stop_flag.set()

        elapsed = time.time() - self.start
        efficiency = self.done / (elapsed / 60) if elapsed > 0 else 0

        print(f"\n成功: {self.done}/{self.total}")
        print(f"失敗: {self.fail}")
        print(f"時間: {elapsed / 3600:.1f}小時")
        print(f"速度: {efficiency:.1f}檔案/分")

    def reader(self, paths):
        for path in paths:
            if self.stop_flag.is_set():
                break

            with self.lock:
                self.active += 1
                self._update()

            try:
                signal, fs = sf.read(path, dtype=np.float32)
                if signal.ndim > 1:
                    signal = signal[:, 0]

                target = int(30 * 60 * fs)
                if len(signal) < target:
                    padded = np.zeros(target, dtype=np.float32)
                    padded[:len(signal)] = signal
                    signal = padded
                elif len(signal) > target:
                    signal = signal[:target]

                self.load_q.put(Task(path, signal, fs))

            except Exception:
                self.write_q.put(('error', path, '讀取失敗'))
                with self.lock:
                    self.active -= 1

        for _ in range(self.num_workers):
            self.load_q.put(None)
        self.reader_finished.set()

    def worker(self):
        # 每個worker有自己的處理器實例（各自的buffer）
        processor = OptimizedProcessor(self.config)
        processed_count = 0

        while True:
            task = self.load_q.get()
            if task is None:
                break

            try:
                result = processor.process(task.signal, task.fs)
                self.write_q.put(('ok', task.path, result))
                processed_count += 1

                # 每6個檔案GC一次，減少GC開銷
                if processed_count % 6 == 0:
                    gc.collect()

            except Exception:
                self.write_q.put(('error', task.path, '處理失敗'))
            finally:
                if hasattr(task, 'signal'):
                    del task.signal
                del task
                if 'result' in locals():
                    del result
                self.load_q.task_done()

        self.write_q.put(('done', None, None))

    def writer(self):
        finished = 0

        while finished < self.num_workers:
            item = self.write_q.get()
            status, path, data = item

            if status == 'done':
                finished += 1
                continue

            try:
                if status == 'ok':
                    out_path = self._get_path(path)
                    np.savetxt(out_path, data, delimiter=',', fmt='%.6f')

                    with self.lock:
                        self.active -= 1
                        self.done += 1
                        self._update()
                else:
                    with self.lock:
                        self.active -= 1
                        self.fail += 1
                        self._update()
            except Exception:
                with self.lock:
                    self.active -= 1
                    self.fail += 1
                    self._update()
            finally:
                if 'data' in locals() and hasattr(data, 'nbytes'):
                    del data
                self.write_q.task_done()

    def _get_path(self, input_path):
        overlap_pct = int(self.config.overlap * 100)
        name = f"{input_path.stem}_PSD_{self.config.window_sec:g}sHannWindow_{overlap_pct}PercentOverlap.csv"

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir / name
        return input_path.parent / name

    def _update(self):
        now = time.time()
        total_done = self.done + self.fail

        if now - self.last_update >= 1.0 or total_done == self.total:
            elapsed = now - self.start

            if elapsed > 0:
                rate = (total_done / elapsed) * 60
                remaining = self.total - total_done
                eta_hours = remaining / (rate * 60) if rate > 0 else 0
                eta = f"ETA: {eta_hours:.1f}h" if eta_hours > 2 else f"ETA: {eta_hours * 60:.0f}min"
            else:
                rate = 0
                eta = "ETA: --"

            progress = total_done / self.total if self.total > 0 else 0
            bar_len = 40
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)

            status = []
            if self.active > 0:
                status.append(f"處理:{self.active}")
            if self.fail > 0:
                status.append(f"失敗:{self.fail}")
            info = f" | {', '.join(status)}" if status else ""

            print(f"\r[{bar}] {progress * 100:.1f}% ({total_done}/{self.total}) "
                  f"| {rate:.1f}/分 | {eta}{info}", end='', flush=True)

            self.last_update = now


def find_files(path, date=None, prefix=None):
    path = Path(path)
    if path.is_file():
        return [path]

    if date:
        if not prefix:
            all_items = os.listdir(str(path))
            wav_files = [f for f in all_items if f.lower().endswith('.wav')]
            prefixes = {f.split('.')[0] for f in wav_files if '.' in f}
            if len(prefixes) == 1:
                prefix = prefixes.pop()
            else:
                raise ValueError(f"多個前綴，指定: {sorted(prefixes)}")

        all_items = os.listdir(str(path))
        files = sorted([path / f for f in all_items
                        if f.lower().endswith('.wav') and f.startswith(f"{prefix}.{date}")])
    else:
        all_items = os.listdir(str(path))
        files = sorted([path / f for f in all_items if f.lower().endswith('.wav')])

    return files


def main():
    parser = argparse.ArgumentParser(description='PAMGuide')
    parser.add_argument('path', type=Path)
    parser.add_argument('--date', '-d')
    parser.add_argument('--prefix', '-p')
    parser.add_argument('--output', '-o', type=Path)
    parser.add_argument('--sensitivity', '-s', type=float)
    parser.add_argument('--time-avg', '-t', type=float)

    args = parser.parse_args()

    kwargs = {}
    if args.sensitivity is not None:
        kwargs['sensitivity'] = args.sensitivity
    if args.time_avg is not None:
        kwargs['time_avg'] = args.time_avg

    config = Config(**kwargs)
    files = find_files(args.path, args.date, args.prefix)

    if not files:
        print("無WAV文件")
        return

    runner = Runner(config, args.output)
    runner.run(files)


if __name__ == '__main__':
    main()