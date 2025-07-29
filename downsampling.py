#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
import os
import numpy as np
import soundfile as sf
import scipy as sp
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    target_fs: float = 64000.0


class SimpleProcessor:
    """簡單一次性降採樣處理器"""

    def __init__(self, config: Config):
        self.config = config

    def process_file(self, input_path: Path, output_path: Path):
        """一次性處理單個檔案"""
        try:
            # 一次性讀取
            signal, fs = sf.read(input_path, dtype=np.float32)

            # 多聲道轉單聲道
            if signal.ndim > 1:
                signal = signal[:, 0]

            # 計算降採樣參數
            factor = int(fs / self.config.target_fs)

            if factor <= 1:
                # 不需要降採樣
                new_fs = fs
                result = signal
            else:
                # 清理信號
                signal = self._clean_signal(signal)

                # 降採樣
                result = sp.signal.decimate(signal, factor, ftype='iir')
                new_fs = fs / factor

            # 一次性寫出
            sf.write(output_path, result, int(new_fs))
            return True

        except Exception as e:
            print(f"處理失敗 {input_path}: {e}")
            return False

    def _clean_signal(self, signal: np.ndarray) -> np.ndarray:
        """清理信號"""
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        signal = np.clip(signal, -10.0, 10.0)
        return signal.astype(np.float32)


class SimpleRunner:
    """簡單的單線程處理器"""

    def __init__(self, config: Config, output_dir: Path = None):
        self.config = config
        self.output_dir = output_dir
        self.processor = SimpleProcessor(config)

    def run(self, files):
        total = len(files)
        success = 0
        failed = 0
        start_time = time.time()

        print(f"文件: {total}")
        print(f"目標採樣率: {int(self.config.target_fs)}Hz")
        print()

        for i, input_path in enumerate(files, 1):
            output_path = self._get_output_path(input_path)

            print(f"[{i}/{total}] {input_path.name} -> {output_path.name}")

            file_start = time.time()
            if self.processor.process_file(input_path, output_path):
                success += 1
                file_time = time.time() - file_start
                print(f"  完成 ({file_time:.1f}s)")
            else:
                failed += 1
                print(f"  失敗")

        elapsed = time.time() - start_time
        print(f"\n=== 結果 ===")
        print(f"成功: {success}/{total}")
        print(f"失敗: {failed}")
        print(f"總時間: {elapsed / 60:.1f}分鐘")
        print(f"平均速度: {success / (elapsed / 60):.1f}檔案/分") if elapsed > 0 else None

    def _get_output_path(self, input_path: Path) -> Path:
        """獲取輸出路徑"""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir / input_path.name
        return input_path.parent / input_path.name


def find_files(path, date=None, prefix=None):
    """查找WAV檔案"""
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
    parser = argparse.ArgumentParser(description='單線程降採樣處理器')
    parser.add_argument('path', type=Path, help='輸入檔案或目錄')
    parser.add_argument('--date', '-d', help='指定日期篩選')
    parser.add_argument('--prefix', '-p', help='指定檔案前綴')
    parser.add_argument('--output', '-o', type=Path, help='輸出目錄')
    parser.add_argument('--target-fs', '-f', type=float, default=64000, help='目標採樣率')

    args = parser.parse_args()

    config = Config(target_fs=args.target_fs)
    files = find_files(args.path, args.date, args.prefix)

    if not files:
        print("找不到WAV檔案")
        return

    runner = SimpleRunner(config, args.output)
    runner.run(files)


if __name__ == '__main__':
    main()