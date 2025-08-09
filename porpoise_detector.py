#!/usr/bin/env python3
"""
Porpoise Click Detection System - Batch Processing
For long-term acoustic monitoring (16 days x 30min files)
"""

import numpy as np
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
import argparse
import pandas as pd
import glob
from tqdm import tqdm


class PorpoiseClickAnalyzer:
    def __init__(self):
        # Basic detection parameters
        self.threshold_factor = 3.5
        self.min_duration = 0.00005  # 50μs
        self.max_duration = 0.00015  # 150μs
        self.min_click_interval = 0.001  # 1ms

        # Group parameters (based on porpoise click train research)
        self.min_clicks_per_group = 6
        self.max_clicks_per_group = 150
        self.max_group_interval = 0.08  # 80ms
        self.min_group_duration = 0.001  # 1ms
        self.max_group_duration = 0.2  # 200ms

        # Signal quality requirements
        self.min_snr_db = 20
        self.min_signal_db = 6

    def analyze_file(self, file_path):
        """Analyze single I/Q file"""
        try:
            # Read I/Q file
            data, sample_rate = sf.read(file_path, dtype=np.float32)
            if data.ndim != 2 or data.shape[1] != 2:
                return None

            iq_data = data[:, 0] + 1j * data[:, 1]

            # Calculate envelope
            envelope = np.abs(iq_data)

            # Adaptive background estimation per file
            background_level = np.percentile(envelope, 5)
            threshold_level = background_level * self.threshold_factor

            # Detection pipeline
            candidates = self._find_candidates(envelope, threshold_level, sample_rate)
            validated_clicks, validation_stats = self._validate_clicks(candidates, iq_data, sample_rate,
                                                                       background_level)
            valid_groups = self._group_and_filter(validated_clicks)

            # Calculate signal statistics
            signal_max = np.max(envelope)
            signal_min = np.min(envelope)
            signal_mean = np.mean(envelope)

            return {
                'filename': Path(file_path).name,
                'valid_groups': valid_groups,
                'background_level': background_level,
                'threshold_level': threshold_level,
                'signal_max': signal_max,
                'signal_min': signal_min,
                'signal_mean': signal_mean,
                'stats': validation_stats
            }

        except Exception:
            return None

    def _find_candidates(self, envelope, threshold_level, sample_rate):
        """Find above-threshold events with valid duration"""
        above_threshold = envelope > threshold_level
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Handle boundaries
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(above_threshold)]])

        # Duration filtering
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
        """Validate signal quality and extract spectral features"""
        validated = []
        last_time = -float('inf')
        context_samples = int(sample_rate * 0.0005)  # 0.5ms context

        # Statistics for parameter tuning (simplified)
        stats = {
            'total_candidates': len(candidates),
            'validated': 0,
            'snr_values': []
        }

        for click in candidates:
            # Check click interval
            if click['start_time'] - last_time < self.min_click_interval:
                continue

            # Calculate signal quality metrics
            signal_amplitude = click['max_amplitude']
            snr_db = 20 * np.log10(signal_amplitude / background_level) if background_level > 0 else 0
            signal_db = 20 * np.log10(signal_amplitude) if signal_amplitude > 0 else -np.inf

            # Record SNR for all candidates
            stats['snr_values'].append(snr_db)

            # Signal quality filtering
            if snr_db < self.min_snr_db or signal_db < self.min_signal_db:
                continue

            # Extract segment for spectral analysis
            start_idx = max(0, click['start'] - context_samples)
            end_idx = min(len(iq_data), click['end'] + context_samples)
            segment = iq_data[start_idx:end_idx]

            if len(segment) < 32:
                continue

            # Spectral analysis (for recording, no filtering)
            try:
                nperseg = min(128, len(segment))
                freqs, psd = signal.welch(segment, sample_rate, nperseg=nperseg)

                # Find peak frequency (for recording only)
                peak_idx = np.argmax(psd)
                peak_freq = abs(freqs[peak_idx])

                # Calculate 3dB bandwidth (for recording only)
                half_max = psd[peak_idx] / 2
                above_half = psd > half_max

                if np.sum(above_half) > 1:
                    freq_indices = np.where(above_half)[0]
                    bandwidth = freqs[freq_indices[-1]] - freqs[freq_indices[0]]
                else:
                    bandwidth = 0

                # Save features (no frequency filtering)
                click.update({
                    'peak_frequency': peak_freq,
                    'bandwidth': bandwidth,
                    'snr_db': snr_db,
                    'signal_db': signal_db
                })
                validated.append(click)
                stats['validated'] += 1
                last_time = click['start_time']

            except Exception:
                # Save basic info if spectral analysis fails
                click.update({
                    'peak_frequency': 0,
                    'bandwidth': 0,
                    'snr_db': snr_db,
                    'signal_db': signal_db
                })
                validated.append(click)
                stats['validated'] += 1
                last_time = click['start_time']

        return validated, stats

    def _group_and_filter(self, validated_clicks):
        """Group validated clicks and filter invalid groups"""
        if len(validated_clicks) < self.min_clicks_per_group:
            return []

        # Sort by time
        clicks_sorted = sorted(validated_clicks, key=lambda x: x['start_time'])

        groups = []
        current_group = [clicks_sorted[0]]

        # Group based on time intervals
        for i in range(1, len(clicks_sorted)):
            interval = clicks_sorted[i]['start_time'] - current_group[-1]['start_time']

            if interval <= self.max_group_interval:
                current_group.append(clicks_sorted[i])
            else:
                valid_group = self._create_group(current_group)
                if valid_group:
                    groups.append(valid_group)
                current_group = [clicks_sorted[i]]

        # Handle last group
        valid_group = self._create_group(current_group)
        if valid_group:
            groups.append(valid_group)

        return groups

    def _create_group(self, clicks):
        """Create group data structure from click list"""
        if len(clicks) < self.min_clicks_per_group:
            return None

        # Calculate group time range
        duration = clicks[-1]['start_time'] - clicks[0]['start_time']

        # Check group duration
        if not (self.min_group_duration <= duration <= self.max_group_duration):
            return None

        # Check group size
        if len(clicks) > self.max_clicks_per_group:
            return None

        # Calculate group statistics
        snr_values = [c.get('snr_db', 0) for c in clicks]
        signal_db_values = [c.get('signal_db', 0) for c in clicks]
        freq_values = [c.get('peak_frequency', 0) for c in clicks if c.get('peak_frequency', 0) > 0]
        bandwidth_values = [c.get('bandwidth', 0) for c in clicks if c.get('bandwidth', 0) > 0]

        return {
            'start_time': clicks[0]['start_time'],
            'end_time': clicks[-1]['start_time'],
            'duration': duration,
            'num_clicks': len(clicks),
            'mean_snr_db': np.mean(snr_values) if snr_values else 0,
            'mean_signal_db': np.mean(signal_db_values) if signal_db_values else 0,
            'mean_frequency_hz': np.mean(freq_values) if freq_values else 0,
            'mean_bandwidth_hz': np.mean(bandwidth_values) if bandwidth_values else 0
        }

    def save_file_results(self, result, output_dir):
        """Save detection results and statistics in separate directories"""
        if not result:
            return None

        filename_stem = Path(result['filename']).stem
        output_dir = Path(output_dir)

        # Create subdirectories
        detection_dir = output_dir / "detections"
        stats_dir = output_dir / "stats"
        detection_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save detection results
        detections_file = detection_dir / f"{filename_stem}.csv"
        csv_data = []

        if result['valid_groups']:
            # Has detections - save group data
            for i, group in enumerate(result['valid_groups']):
                csv_data.append({
                    'group_id': i + 1,
                    'start_time': group['start_time'],
                    'end_time': group['end_time'],
                    'duration_ms': group['duration'] * 1000,
                    'num_clicks': group['num_clicks'],
                    'mean_snr_db': group['mean_snr_db'],
                    'mean_signal_db': group['mean_signal_db'],
                    'mean_frequency_hz': group['mean_frequency_hz'],
                    'mean_bandwidth_hz': group['mean_bandwidth_hz']
                })
        else:
            # No detections - save empty row
            csv_data.append({
                'group_id': 0,
                'start_time': 0,
                'end_time': 0,
                'duration_ms': 0,
                'num_clicks': 0,
                'mean_snr_db': 0,
                'mean_signal_db': 0,
                'mean_frequency_hz': 0,
                'mean_bandwidth_hz': 0
            })

        df_detections = pd.DataFrame(csv_data)
        df_detections.to_csv(detections_file, index=False)

        # 2. Save statistics for parameter tuning
        stats_file = stats_dir / f"{filename_stem}.csv"
        stats = result['stats']

        stats_data = [{
            'filename': result['filename'],
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

        df_stats = pd.DataFrame(stats_data)
        df_stats.to_csv(stats_file, index=False)

        return str(detections_file)

    def batch_process(self, input_path, output_dir):
        """Batch process multiple files from pattern or directory"""
        # Handle different input types
        input_path = Path(input_path)

        if input_path.is_dir():
            # Input is a directory - find all WAV files
            files = sorted(input_path.glob("*.wav"))
            print(f"Found {len(files)} WAV files in directory: {input_path}")
        else:
            # Input is a pattern - use glob
            files = sorted(glob.glob(str(input_path)))
            print(f"Found {len(files)} files matching pattern: {input_path}")

        if not files:
            print(f"No WAV files found")
            return

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics variables
        total_detections = 0
        processed_files = 0
        files_with_detections = 0

        # Batch process files with progress bar
        for file_path in tqdm(files, desc="Processing files"):
            result = self.analyze_file(file_path)

            if result:
                # Always save results (even if no detections)
                output_path = self.save_file_results(result, output_dir)
                if output_path:
                    processed_files += 1
                    if result['valid_groups']:
                        total_detections += len(result['valid_groups'])
                        files_with_detections += 1

        # Print processing summary
        print(f"\nProcessed {processed_files}/{len(files)} files")
        print(f"Files with detections: {files_with_detections}")
        print(f"Total detections: {total_detections}")
        if files_with_detections > 0:
            print(f"Average detections per file (with detections): {total_detections / files_with_detections:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Porpoise Click Detection System - Batch Processing')
    parser.add_argument('input', help='Input directory or file pattern (e.g., "/path/to/files", "*.wav")')
    parser.add_argument('--output-dir', '-o', default='./results',
                        help='Output directory (default: ./results)')

    args = parser.parse_args()

    # Run batch analysis
    analyzer = PorpoiseClickAnalyzer()
    analyzer.batch_process(args.input, args.output_dir)


if __name__ == '__main__':
    main()