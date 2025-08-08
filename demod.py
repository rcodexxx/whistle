#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as signal
from tqdm import tqdm


class DemodConfig:
    low_freq = 115000.0
    high_freq = 145000.0
    target_fs = (high_freq - low_freq) * 2.5  # 75kHz


class DemodProcessor:
    def __init__(self, config: DemodConfig):
        self.config = config
        self.center_freq = (config.low_freq + config.high_freq) / 2  # 130kHz
        self.cutoff = (config.high_freq - config.low_freq) / 2 * 1.1  # 16.5kHz

    def process(self, signal_data: np.ndarray, fs: float) -> np.ndarray:
        # Design filters
        sos_bp = signal.butter(6, [self.config.low_freq, self.config.high_freq],
                               btype='band', fs=fs, output='sos')
        sos_lp = signal.butter(4, self.cutoff, btype='low', fs=fs, output='sos')

        # Bandpass filter
        bandpassed = signal.sosfiltfilt(sos_bp, signal_data)

        # Frequency mixing demod
        t = np.arange(len(bandpassed)) / fs
        lo_signal = np.exp(-1j * 2 * np.pi * self.center_freq * t, dtype=np.complex64)
        baseband = bandpassed * lo_signal

        # Lowpass filter
        filtered = signal.sosfiltfilt(sos_lp, baseband)

        # Resample: 384kHz -> 75kHz
        up = self.config.target_fs
        down = fs
        result = signal.resample_poly(filtered, up, down)

        return result.astype(np.complex64)


def get_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    try:
        if input_root.is_file():
            return output_root / input_path.with_suffix('.wav').name
        else:
            relative_path = input_path.relative_to(input_root)
            return output_root / relative_path.with_suffix('.wav')
    except ValueError:
        return output_root / input_path.with_suffix('.wav').name


def find_wav_files(path: Path) -> list:
    if path.is_file():
        return [path] if path.suffix.lower() == '.wav' else []
    return sorted(path.glob('**/*.wav'))


def process_single_file(input_path: Path, config: DemodConfig, input_root: Path, output_root: Path):
    # Read file
    signal_data, fs = sf.read(input_path, dtype=np.float32)
    if signal_data.ndim > 1:
        signal_data = signal_data[:, 0]  # Use first channel

    if len(signal_data) == 0:
        raise ValueError("Empty audio data")

    # Process signal
    processor = DemodProcessor(config)
    result = processor.process(signal_data, fs)

    # Write output
    output_path = get_output_path(input_path, input_root, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    iq_interleaved = np.stack([np.real(result), np.imag(result)], axis=1)
    sf.write(str(output_path), iq_interleaved, int(config.target_fs), subtype='FLOAT')

    # Verify output
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise OSError("Output file write failed")


def main():
    parser = argparse.ArgumentParser(description='Demodulator')
    parser.add_argument('--input', '-i', type=Path, required=True, help='Input directory or file')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output directory')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input path not found: {args.input}")
        sys.exit(1)

    config = DemodConfig()
    files = find_wav_files(args.input)

    if not files:
        print("No WAV files found")
        sys.exit(1)

    print(f"Found {len(files)} files")
    print(
        f"Processing: {config.low_freq / 1000:.1f}-{config.high_freq / 1000:.1f} kHz -> {config.target_fs / 1000:.1f} kHz")

    # Batch process
    file_path = None
    try:
        for file_path in tqdm(files, desc="Progress"):
            process_single_file(file_path, config, args.input, args.output)

        print(f"\nProcessed {len(files)} files successfully")

    except Exception as e:
        if file_path:
            print(f"\nFailed processing: {file_path.name}")
        else:
            print(f"\nProcessing failed")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()