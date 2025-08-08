#!/usr/bin/env python3
import argparse
from pathlib import Path
import gc
import datetime as dt
import numpy as np
import soundfile as sf
import scipy.signal as signal
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# --- 組態與核心處理類別 (未修改) ---
class DemodConfig:
    low_freq: float = 115000.0
    high_freq: float = 145000.0
    target_fs: float = (high_freq - low_freq) * 2.5


class DemodProcessor:
    def __init__(self, config: DemodConfig):
        self.config = config
        self.center_freq = (config.low_freq + config.high_freq) / 2
        self.cutoff = (config.high_freq - config.low_freq) / 2 * 1.1

    def process(self, signal_data: np.ndarray, fs: float) -> np.ndarray:
        up = int(self.config.target_fs)
        down = int(fs)
        sos_bp = signal.butter(6, [self.config.low_freq, self.config.high_freq], btype='band', fs=fs, output='sos')
        sos_lp = signal.butter(4, self.cutoff, btype='low', fs=fs, output='sos')
        chunk_size = int(fs * 60)
        num_chunks = int(np.ceil(len(signal_data) / chunk_size))
        zi_bp = signal.sosfilt_zi(sos_bp)
        zi_lp = signal.sosfilt_zi(sos_lp)
        output_chunks = []
        for i in range(num_chunks):
            chunk = signal_data[i * chunk_size: (i + 1) * chunk_size]
            if chunk.size == 0: continue
            bandpassed, zi_bp = signal.sosfilt(sos_bp, chunk, zi=zi_bp)
            t = (np.arange(len(bandpassed)) + i * chunk_size) / fs
            lo_signal = np.exp(-1j * 2 * np.pi * self.center_freq * t, dtype=np.complex64)
            baseband = bandpassed * lo_signal
            filtered, zi_lp = signal.sosfilt(sos_lp, baseband, zi=zi_lp)
            downsampled_chunk = signal.resample_poly(filtered, up, down)
            output_chunks.append(downsampled_chunk)
        if not output_chunks:
            return np.array([], dtype=np.complex64)
        final_result = np.concatenate(output_chunks).astype(np.complex64)
        return final_result


# --- 已修改: 新的 WavWriter 取代舊的 SigMFWriter ---
class WavWriter:
    def __init__(self, config: DemodConfig):
        self.config = config

    def _get_output_path(self, input_path: Path, input_root: Path, output_root: Path) -> Path:
        try:
            if input_root.is_file():
                return output_root / input_path.with_suffix('.wav').name
            else:
                relative_path = input_path.relative_to(input_root)
                return output_root / relative_path.with_suffix('.wav')
        except ValueError:
            return output_root / input_path.with_suffix('.wav').name

    def write(self, data: np.ndarray, output_path: Path, original_path: Path, original_fs: float):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not np.iscomplexobj(data):
            raise ValueError(f"Expected complex data, got {data.dtype}")

        iq_interleaved = np.stack([np.real(data), np.imag(data)], axis=1)
        sf.write(str(output_path), iq_interleaved, int(self.config.target_fs), subtype='FLOAT')


# --- 多工處理核心函數 ---
def process_single_file(args_tuple):
    input_path, config, input_root, output_root = args_tuple
    try:
        processor = DemodProcessor(config)
        writer = WavWriter(config)

        signal_data, fs = sf.read(input_path, dtype=np.float32)
        if signal_data.ndim > 1:
            signal_data = signal_data[:, 0]

        result = processor.process(signal_data, fs)

        output_path = writer._get_output_path(input_path, input_root, output_root)
        writer.write(result, output_path, input_path, fs)

        del signal_data, result
        gc.collect()

        return (str(input_path), "Success")
    except Exception as e:
        error_info = f"{type(e).__name__}: {e}"
        return (str(input_path), error_info)


def find_wav_files(path):
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() == '.wav' else []
    print(f"掃描目錄: {path}")
    wav_files = sorted(path.glob('**/*.wav'))
    print(f"找到 {len(wav_files)} 個 WAV 檔案")
    return wav_files


# --- 主程式入口 (未修改) ---
def main():
    parser = argparse.ArgumentParser(
        description='Underwater acoustic signal demodulation processing (Parallel Version)')
    parser.add_argument('--input', '-i', type=Path, required=True, help='Input directory or file')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output directory')
    parser.add_argument('--jobs', '-j', type=int, default=None,
                        help='Number of parallel jobs. Defaults to all available CPU cores.')
    args = parser.parse_args()
    if not args.input.exists():
        print(f"輸入路徑不存在: {args.input}")
        return
    config = DemodConfig()
    files = find_wav_files(args.input)
    if not files:
        print("找不到任何 WAV 檔案")
        return
    if args.jobs:
        num_processes = min(args.jobs, cpu_count())
    else:
        num_processes = cpu_count()
    print(f"準備使用 {num_processes} 個 CPU 核心處理 {len(files)} 個檔案...")
    tasks = [(path, config, args.input, args.output) for path in files]
    success_count = 0
    failed_files = []
    with Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(process_single_file, tasks)
        for result in tqdm(results_iterator, total=len(files), desc="Processing files"):
            path, status = result
            if status == "Success":
                success_count += 1
            else:
                failed_files.append((path, status))
    print("\n--- 全部處理完成 ---")
    print(f"成功: {success_count} / {len(files)}")
    if failed_files:
        print(f"失敗: {len(failed_files)} / {len(files)}")
        print("失敗檔案詳情:")
        for filename, error in failed_files:
            print(f"  - {Path(filename).name}: {error}")


if __name__ == '__main__':
    main()