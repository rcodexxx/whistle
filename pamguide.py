#!/usr/bin/env python3

import argparse
import time
import queue
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import psutil
import os

import numpy as np
import soundfile as sf
import scipy as sp


@dataclass(frozen=True)
class Config:
    sensitivity: float = -176.3
    window_sec: float = 1.0
    overlap: float = 0.5
    freq_range: Tuple[float, float] = (1.0, 96000.0)
    time_avg: float = 300.0


@dataclass
class Task:
    path: Path
    signal: np.ndarray
    fs: float


class Processor:
    def __init__(self, config: Config):
        self.config = config
        self._pref = 1.0

    def process(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """使用 SciPy 及向量化操作處理信號，產生 PSD 矩陣。"""
        window_samples = int(self.config.window_sec * fs)
        overlap_samples = int(np.ceil(window_samples * self.config.overlap))

        # 核心頻譜計算
        freqs, times, sxx = sp.signal.spectrogram(
            signal,
            fs=fs,
            window='hann',
            nperseg=window_samples,
            noverlap=overlap_samples,
            scaling='density',
            mode='psd'
        )

        # 在線性尺度上進行頻率過濾，以提升效率
        freq_mask = (freqs >= self.config.freq_range[0]) & (freqs <= self.config.freq_range[1])
        filtered_freqs = freqs[freq_mask]
        filtered_sxx = sxx[freq_mask, :]

        # 轉換為 dB 並應用敏感度校正
        psd_db = 10 * np.log10(filtered_sxx.T / (self._pref ** 2)) - self.config.sensitivity

        # 時間平均
        final_psd = psd_db
        final_times = times
        if self.config.time_avg > 0 and len(times) > 1:
            step = times[1] - times[0]
            final_psd, final_times = self._welch_avg(psd_db, times, step)

        # 建立輸出矩陣
        return self._build_matrix(final_psd, filtered_freqs, final_times)

    def _welch_avg(self, psd_db: np.ndarray, original_times: np.ndarray, step: float):
        """向量化時間平均，取代 for 迴圈以提升效能。"""
        if step == 0:
            return psd_db, original_times

        factor = int(round(self.config.time_avg / step))
        if factor <= 1:
            return psd_db, original_times

        # 轉為線性功率以進行正確的物理量平均
        linear_power = 10 ** (psd_db / 10)
        n_rows, n_cols = linear_power.shape

        # 計算輸出點數，並為最後不完整的數據塊準備填充
        n_out = int(np.ceil(n_rows / factor))

        # 填充數據，使總行數可被平均因子整除
        target_rows = n_out * factor
        padding_needed = target_rows - n_rows
        if padding_needed > 0:
            # 使用 NaN 填充，以便 nanmean 正確處理不完整的數據塊
            padding = np.full((padding_needed, n_cols), np.nan, dtype=np.float32)
            linear_power = np.vstack([linear_power, padding])

        # 向量化平均：reshape 將數據分組，nanmean 沿指定軸計算平均
        reshaped_power = linear_power.reshape(n_out, factor, n_cols)
        avg_power = np.nanmean(reshaped_power, axis=1)

        # 轉回 dB 尺度
        avg_psd_db = 10 * np.log10(avg_power)

        # 計算對應的時間軸
        new_times = original_times[::factor][:n_out]

        return avg_psd_db.astype(np.float32), new_times.astype(np.float32)

    def _build_matrix(self, psd: np.ndarray, freqs: np.ndarray, times: np.ndarray) -> np.ndarray:
        """將結果組合成最終的輸出矩陣格式。"""
        rows, cols = psd.shape
        matrix = np.zeros((rows + 1, cols + 1), dtype=np.float32)
        matrix[0, 0] = 2211
        matrix[0, 1:] = freqs
        matrix[1:, 0] = times
        matrix[1:, 1:] = psd
        return matrix


class Runner:
    def __init__(self, config: Config, output_dir: Optional[Path] = None):
        self.config = config
        self.output_dir = output_dir
        self.processor = Processor(config)

        self.load_q = queue.Queue(maxsize=2)
        self.write_q = queue.Queue(maxsize=4)

        self.done = 0
        self.fail = 0
        self.active = 0
        self.total = 0
        self.start = 0
        self.last_update = 0
        self.lock = threading.Lock()

    def reader(self, paths):
        for path in paths:
            with self.lock:
                self.active += 1
                self._update()

            try:
                signal, fs = sf.read(path, dtype=np.float32)
                if signal.ndim > 1:
                    signal = signal[:, 0]

                # 固定30分鐘
                target = int(30 * 60 * fs)
                if len(signal) < target:
                    padded = np.zeros(target, dtype=np.float32)
                    padded[:len(signal)] = signal
                    signal = padded
                elif len(signal) > target:
                    signal = signal[:target]

                self.load_q.put(Task(path, signal, fs))
            except Exception as e:
                self.write_q.put(('error', path, str(e)))
                with self.lock:
                    self.active -= 1

        for _ in range(6):
            self.load_q.put(None)

    def worker(self):
        while True:
            task = self.load_q.get()
            if task is None:
                break

            try:
                result = self.processor.process(task.signal, task.fs)
                self.write_q.put(('ok', task.path, result))
            except Exception as e:
                self.write_q.put(('error', task.path, str(e)))
            finally:
                self.load_q.task_done()

        self.write_q.put(('done', None, None))

    def writer(self):
        finished = 0
        while finished < 6:
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
                eta = f"ETA: {remaining / rate:.1f}min" if rate > 0 else "ETA: --"
            else:
                rate = 0
                eta = "ETA: --"

            progress = total_done / self.total
            bar_len = 40
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)

            mem = psutil.virtual_memory().available / (1024 ** 3)

            status = []
            if self.active > 0:
                status.append(f"處理:{self.active}")
            if self.fail > 0:
                status.append(f"失敗:{self.fail}")
            info = f" | {', '.join(status)}" if status else ""

            print(f"\r[{bar}] {progress * 100:.1f}% ({total_done}/{self.total}) "
                  f"| {rate:.1f}/分 | {eta} | {mem:.1f}GB{info}", end='', flush=True)

            self.last_update = now

    def run(self, paths):
        self.total = len(paths)
        self.start = time.time()

        print(f"文件: {self.total}")
        print(f"敏感度: {self.config.sensitivity}dB")
        print(f"時間平均: {self.config.time_avg}s")
        print(f"內存: {psutil.virtual_memory().available / (1024 ** 3):.1f}GB")
        print()

        self._update()

        threads = [
            threading.Thread(target=self.reader, args=(paths,)),
            *[threading.Thread(target=self.worker) for _ in range(6)],
            threading.Thread(target=self.writer)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        elapsed = time.time() - self.start
        print(f"\n完成: {self.done}/{self.total} | {elapsed / 60:.1f}分 | {self.done / (elapsed / 60):.1f}/分")

        if self.fail > 0:
            print(f"失敗: {self.fail}")


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
    parser = argparse.ArgumentParser(description='PAMGuide生產')
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