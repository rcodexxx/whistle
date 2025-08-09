#!/usr/bin/env python3
"""
Daily Porpoise Detection Analysis and Visualization
Analyze and visualize one day of porpoise detection results
ENHANCED VERSION - includes signal dB range and additional statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import glob


class DailyAnalyzer:
    def __init__(self):
        self.detection_files = []
        self.stats_files = []
        self.detections_df = None
        self.stats_df = None

    def load_data(self, results_dir):
        """Load detection and stats data from results directory"""
        results_path = Path(results_dir)

        # Load detection files
        detection_dir = results_path / "detections"
        stats_dir = results_path / "stats"

        if not detection_dir.exists() or not stats_dir.exists():
            raise ValueError(f"Results directory must contain 'detections' and 'stats' subdirectories")

        # Find all CSV files
        detection_files = sorted(detection_dir.glob("*.csv"))
        stats_files = sorted(stats_dir.glob("*.csv"))

        print(f"Found {len(detection_files)} detection files")
        print(f"Found {len(stats_files)} stats files")

        # Load and combine detection data
        all_detections = []
        for file_path in detection_files:
            try:
                df = pd.read_csv(file_path)
                if not df.empty and df['group_id'].iloc[0] != 0:  # Skip empty detection files
                    df['source_file'] = file_path.stem
                    all_detections.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Load and combine stats data
        all_stats = []
        for file_path in stats_files:
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = file_path.stem
                all_stats.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Combine data
        self.detections_df = pd.concat(all_detections, ignore_index=True) if all_detections else pd.DataFrame()
        self.stats_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

        # Add dB calculations
        self._calculate_db_values()

        # Extract time information from filenames
        self._extract_time_info()

        # Debug info
        print(f"Loaded detections: {len(self.detections_df)} rows")
        print(f"Loaded stats: {len(self.stats_df)} rows")

    def _calculate_db_values(self):
        """Calculate dB values from signal levels"""
        if not self.stats_df.empty:
            # Calculate dB values, handling zero/negative values
            self.stats_df['signal_max_db'] = np.where(
                self.stats_df['signal_max'] > 0,
                20 * np.log10(self.stats_df['signal_max']),
                -np.inf
            )
            self.stats_df['signal_min_db'] = np.where(
                self.stats_df['signal_min'] > 0,
                20 * np.log10(self.stats_df['signal_min']),
                -np.inf
            )
            self.stats_df['signal_mean_db'] = np.where(
                self.stats_df['signal_mean'] > 0,
                20 * np.log10(self.stats_df['signal_mean']),
                -np.inf
            )
            self.stats_df['background_level_db'] = np.where(
                self.stats_df['background_level'] > 0,
                20 * np.log10(self.stats_df['background_level']),
                -np.inf
            )

            # Calculate dynamic range for each file
            self.stats_df['dynamic_range_db'] = self.stats_df['signal_max_db'] - self.stats_df['signal_min_db']

    def _extract_time_info(self):
        """Extract time information from filenames"""
        if not self.stats_df.empty:
            # Try to extract time from filename - this is dataset specific
            # You may need to adjust this based on your filename format
            self.stats_df['file_hour'] = range(len(self.stats_df))  # Placeholder - use file order as time

        if not self.detections_df.empty:
            # Add time info to detections
            file_to_hour = dict(zip(self.stats_df['source_file'], self.stats_df['file_hour']))
            self.detections_df['file_hour'] = self.detections_df['source_file'].map(file_to_hour)

    def generate_daily_summary(self):
        """Generate daily summary statistics - ENHANCED VERSION"""
        if self.stats_df.empty:
            return "No data to analyze"

        total_files = len(self.stats_df)
        files_with_detections = len(self.detections_df['source_file'].unique()) if not self.detections_df.empty else 0
        total_groups = len(self.detections_df) if not self.detections_df.empty else 0
        total_clicks = self.detections_df['num_clicks'].sum() if not self.detections_df.empty else 0

        # Signal quality stats - safe calculations
        avg_background = self.stats_df['background_level'].mean() if 'background_level' in self.stats_df.columns else 0
        avg_snr = self.stats_df['snr_mean'].mean() if 'snr_mean' in self.stats_df.columns and not self.stats_df[
            'snr_mean'].isna().all() else 0
        detection_rate = (files_with_detections / total_files * 100) if total_files > 0 else 0

        # Safe division for average clicks per group
        avg_clicks_per_group = total_clicks / total_groups if total_groups > 0 else 0

        # Enhanced signal statistics
        signal_stats = ""
        if 'signal_max_db' in self.stats_df.columns:
            valid_max_db = self.stats_df['signal_max_db'][self.stats_df['signal_max_db'] != -np.inf]
            valid_min_db = self.stats_df['signal_min_db'][self.stats_df['signal_min_db'] != -np.inf]
            valid_mean_db = self.stats_df['signal_mean_db'][self.stats_df['signal_mean_db'] != -np.inf]
            valid_bg_db = self.stats_df['background_level_db'][self.stats_df['background_level_db'] != -np.inf]
            valid_dynamic_range = self.stats_df['dynamic_range_db'][np.isfinite(self.stats_df['dynamic_range_db'])]

            if len(valid_max_db) > 0:
                signal_stats = f"""
Signal Level Statistics (dB):
• Maximum Signal: {valid_max_db.max():.1f} dB
• Minimum Signal: {valid_min_db.min():.1f} dB
• Average Maximum: {valid_max_db.mean():.1f} dB
• Average Minimum: {valid_min_db.mean():.1f} dB
• Average Mean Signal: {valid_mean_db.mean():.1f} dB
• Average Background: {valid_bg_db.mean():.1f} dB
• Average Dynamic Range: {valid_dynamic_range.mean():.1f} dB"""

        # Frequency analysis for detections
        frequency_stats = ""
        if not self.detections_df.empty and 'mean_frequency_hz' in self.detections_df.columns:
            valid_freq = self.detections_df['mean_frequency_hz'][self.detections_df['mean_frequency_hz'] > 0]
            if len(valid_freq) > 0:
                frequency_stats = f"""
Frequency Analysis:
• Mean Frequency: {valid_freq.mean() / 1000:.1f} kHz
• Frequency Range: {valid_freq.min() / 1000:.1f} - {valid_freq.max() / 1000:.1f} kHz
• Frequency Std: {valid_freq.std() / 1000:.1f} kHz"""

        summary = f"""
=== Daily Porpoise Detection Summary ===
Analysis Period: {total_files} files (approximately {total_files * 0.5:.1f} hours)

Detection Results:
• Total Groups Detected: {total_groups}
• Total Clicks: {total_clicks}
• Files with Detections: {files_with_detections}/{total_files} ({detection_rate:.1f}%)
• Average Clicks per Group: {avg_clicks_per_group:.1f}

Signal Quality:
• Average Background Level: {avg_background:.6f}
• Average SNR: {avg_snr:.1f} dB
• Detection Rate: {detection_rate:.1f}%{signal_stats}{frequency_stats}

Peak Activity:
"""

        if not self.detections_df.empty and 'file_hour' in self.detections_df.columns:
            hourly_counts = self.detections_df.groupby('file_hour').size()
            if not hourly_counts.empty:
                peak_hour = hourly_counts.idxmax()
                peak_count = hourly_counts.max()
                summary += f"• Peak Activity: File #{peak_hour} ({peak_count} groups)\n"
            else:
                summary += "• No peak activity data available\n"
        else:
            summary += "• No activity detected\n"

        return summary

    def create_visualizations(self, output_dir):
        """Create comprehensive visualization plots - ENHANCED VERSION"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with subplots
        fig = plt.figure(figsize=(24, 16))

        # 1. Timeline of detections (top plot)
        ax1 = plt.subplot(3, 4, (1, 4))  # Span across top
        self._plot_detection_timeline(ax1)

        # 2. Signal dB distribution
        ax2 = plt.subplot(3, 4, 5)
        self._plot_signal_db_distribution(ax2)

        # 3. Signal quality distribution
        ax3 = plt.subplot(3, 4, 6)
        self._plot_signal_quality(ax3)

        # 4. Detection statistics
        ax4 = plt.subplot(3, 4, 7)
        self._plot_detection_stats(ax4)

        # 5. Hourly activity heatmap
        ax5 = plt.subplot(3, 4, 8)
        self._plot_activity_heatmap(ax5)

        # 6. Signal vs Background comparison
        ax6 = plt.subplot(3, 4, 9)
        self._plot_signal_vs_background(ax6)

        # 7. Dynamic range analysis
        ax7 = plt.subplot(3, 4, 10)
        self._plot_dynamic_range(ax7)

        # 8. Frequency distribution (if available)
        ax8 = plt.subplot(3, 4, 11)
        self._plot_frequency_distribution(ax8)

        # 9. Detection quality scatter
        ax9 = plt.subplot(3, 4, 12)
        self._plot_detection_quality_scatter(ax9)

        plt.tight_layout()

        # Save plot
        plot_path = output_path / "enhanced_daily_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Enhanced visualization saved to: {plot_path}")

        # Create additional detailed plots only if we have detection data
        if not self.detections_df.empty:
            self._create_detailed_plots(output_path)

    def _plot_signal_db_distribution(self, ax):
        """Plot signal dB level distribution"""
        if self.stats_df.empty or 'signal_max_db' not in self.stats_df.columns:
            ax.text(0.5, 0.5, 'No dB data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Signal dB Distribution')
            return

        # Get valid dB values
        valid_max_db = self.stats_df['signal_max_db'][self.stats_df['signal_max_db'] != -np.inf]
        valid_min_db = self.stats_df['signal_min_db'][self.stats_df['signal_min_db'] != -np.inf]
        valid_mean_db = self.stats_df['signal_mean_db'][self.stats_df['signal_mean_db'] != -np.inf]

        if len(valid_max_db) > 0:
            ax.hist(valid_max_db, bins=30, alpha=0.7, label='Max dB', color='red')
            ax.hist(valid_mean_db, bins=30, alpha=0.7, label='Mean dB', color='blue')
            ax.hist(valid_min_db, bins=30, alpha=0.7, label='Min dB', color='green')

            ax.set_xlabel('Signal Level (dB)')
            ax.set_ylabel('Frequency')
            ax.set_title('Signal dB Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid dB data', ha='center', va='center', transform=ax.transAxes)

    def _plot_signal_vs_background(self, ax):
        """Plot signal vs background levels"""
        if self.stats_df.empty or 'signal_mean_db' not in self.stats_df.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Signal vs Background')
            return

        valid_signal = self.stats_df['signal_mean_db'][self.stats_df['signal_mean_db'] != -np.inf]
        valid_bg = self.stats_df['background_level_db'][self.stats_df['background_level_db'] != -np.inf]

        if len(valid_signal) > 0 and len(valid_bg) > 0:
            ax.scatter(valid_bg, valid_signal, alpha=0.6)
            ax.set_xlabel('Background Level (dB)')
            ax.set_ylabel('Mean Signal Level (dB)')
            ax.set_title('Signal vs Background Comparison')

            # Add diagonal line
            min_val = min(valid_bg.min(), valid_signal.min())
            max_val = max(valid_bg.max(), valid_signal.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Equal line')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_dynamic_range(self, ax):
        """Plot dynamic range analysis"""
        if self.stats_df.empty or 'dynamic_range_db' not in self.stats_df.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Dynamic Range')
            return

        valid_range = self.stats_df['dynamic_range_db'][np.isfinite(self.stats_df['dynamic_range_db'])]

        if len(valid_range) > 0:
            ax.hist(valid_range, bins=20, alpha=0.7, color='purple')
            ax.set_xlabel('Dynamic Range (dB)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Dynamic Range Distribution\nMean: {valid_range.mean():.1f} dB')
            ax.grid(True, alpha=0.3)

    def _plot_frequency_distribution(self, ax):
        """Plot frequency distribution for detections"""
        if self.detections_df.empty or 'mean_frequency_hz' not in self.detections_df.columns:
            ax.text(0.5, 0.5, 'No frequency data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Frequency Distribution')
            return

        valid_freq = self.detections_df['mean_frequency_hz'][
                         self.detections_df['mean_frequency_hz'] > 0] / 1000  # Convert to kHz

        if len(valid_freq) > 0:
            ax.hist(valid_freq, bins=20, alpha=0.7, color='orange')
            ax.set_xlabel('Frequency (kHz)')
            ax.set_ylabel('Number of Groups')
            ax.set_title(f'Detection Frequency Distribution\nMean: {valid_freq.mean():.1f} kHz')
            ax.grid(True, alpha=0.3)

    def _plot_detection_quality_scatter(self, ax):
        """Plot detection quality scatter plot"""
        if self.detections_df.empty:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Detection Quality')
            return

        if 'mean_snr_db' in self.detections_df.columns and 'num_clicks' in self.detections_df.columns:
            scatter = ax.scatter(self.detections_df['mean_snr_db'],
                                 self.detections_df['num_clicks'],
                                 alpha=0.6, c=self.detections_df['duration_ms'],
                                 cmap='viridis')
            ax.set_xlabel('Mean SNR (dB)')
            ax.set_ylabel('Number of Clicks')
            ax.set_title('Detection Quality: SNR vs Click Count')
            plt.colorbar(scatter, ax=ax, label='Duration (ms)')
            ax.grid(True, alpha=0.3)

    def _plot_detection_timeline(self, ax):
        """Plot detection timeline across the day - ENHANCED VERSION"""
        if self.stats_df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Detection Timeline')
            return

        # Create hourly detection counts
        if not self.detections_df.empty and 'file_hour' in self.detections_df.columns:
            hourly_groups = self.detections_df.groupby('file_hour').size()
            hours = hourly_groups.index
            counts = hourly_groups.values

            # Bar plot for detections
            bars = ax.bar(hours, counts, alpha=0.7, color='lightblue', label='Groups Detected')

            # Add trend line only if we have enough data points
            if len(hours) > 1:
                try:
                    z = np.polyfit(hours, counts, 1)
                    p = np.poly1d(z)
                    ax.plot(hours, p(hours), "r--", alpha=0.8, label='Trend')
                except:
                    pass

        # Plot background noise level as secondary y-axis
        ax2 = ax.twinx()
        if 'background_level_db' in self.stats_df.columns and 'file_hour' in self.stats_df.columns:
            valid_bg_db = self.stats_df['background_level_db'][self.stats_df['background_level_db'] != -np.inf]
            valid_hours = self.stats_df['file_hour'][self.stats_df['background_level_db'] != -np.inf]

            if len(valid_bg_db) > 0:
                ax2.plot(valid_hours, valid_bg_db, 'g-', alpha=0.6, linewidth=2, label='Background (dB)')

        ax.set_xlabel('File Number (Time Sequence)')
        ax.set_ylabel('Number of Groups', color='blue')
        ax2.set_ylabel('Background Level (dB)', color='green')
        ax.set_title('Detection Timeline and Background Noise')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plot_signal_quality(self, ax):
        """Plot signal quality distributions - ENHANCED VERSION"""
        if self.stats_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Signal Quality')
            return

        # Create box plot for signal quality metrics
        data_to_plot = []
        labels = []

        if 'snr_mean' in self.stats_df.columns:
            snr_data = self.stats_df['snr_mean'].dropna()
            if not snr_data.empty:
                data_to_plot.append(snr_data)
                labels.append('SNR (dB)')

        if 'validated_clicks' in self.stats_df.columns:
            clicks_data = self.stats_df['validated_clicks'].dropna()
            if not clicks_data.empty:
                data_to_plot.append(clicks_data)
                labels.append('Valid Clicks')

        if data_to_plot:
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen']
            for i, patch in enumerate(box_plot['boxes']):
                if i < len(colors):
                    patch.set_facecolor(colors[i])
        else:
            ax.text(0.5, 0.5, 'No signal quality data', ha='center', va='center', transform=ax.transAxes)

        ax.set_title('Signal Quality Distribution')
        ax.grid(True, alpha=0.3)

    def _plot_detection_stats(self, ax):
        """Plot detection statistics - FIXED VERSION"""
        if self.stats_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Detection Stats')
            return

        # Detection success rate
        files_with_detections = len(self.detections_df['source_file'].unique()) if not self.detections_df.empty else 0
        total_files = len(self.stats_df)
        files_without = total_files - files_with_detections

        # Pie chart
        sizes = [files_with_detections, files_without]
        labels = [f'With Detections\n({files_with_detections})', f'No Detections\n({files_without})']
        colors = ['lightgreen', 'lightcoral']

        if total_files > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        else:
            ax.text(0.5, 0.5, 'No files processed', ha='center', va='center', transform=ax.transAxes)

        ax.set_title('Detection Success Rate')

    def _plot_activity_heatmap(self, ax):
        """Plot activity heatmap - FIXED VERSION"""
        if self.detections_df.empty:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Activity Heatmap')
            return

        # Create activity matrix (simplified for demo)
        if 'file_hour' in self.detections_df.columns and 'num_clicks' in self.detections_df.columns:
            hourly_activity = self.detections_df.groupby('file_hour')['num_clicks'].sum()

            # Reshape into a matrix (e.g., 6x8 for 48 files)
            n_files = len(self.stats_df)
            rows = 6
            cols = int(np.ceil(n_files / rows)) if n_files > 0 else 1

            activity_matrix = np.zeros((rows, cols))
            for hour, clicks in hourly_activity.items():
                if hour < n_files:
                    row = hour // cols
                    col = hour % cols
                    if row < rows and col < cols:
                        activity_matrix[row, col] = clicks

            # Create heatmap
            im = ax.imshow(activity_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_title('Activity Heatmap (Clicks per Time Block)')
            ax.set_xlabel('Time Block')
            ax.set_ylabel('Period')

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Total Clicks')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Activity Heatmap')

    def _create_detailed_plots(self, output_path):
        """Create additional detailed plots - ENHANCED VERSION"""

        # 1. Detailed timeline plot
        if not self.detections_df.empty and all(
                col in self.detections_df.columns for col in ['file_hour', 'start_time', 'num_clicks', 'mean_snr_db']):
            plt.figure(figsize=(15, 8))

            # Plot individual detection events
            for _, detection in self.detections_df.iterrows():
                plt.scatter(detection['file_hour'], detection['start_time'],
                            s=detection['num_clicks'] * 10, alpha=0.6,
                            c=detection['mean_snr_db'], cmap='viridis')

            plt.colorbar(label='Mean SNR (dB)')
            plt.xlabel('File Number')
            plt.ylabel('Detection Time within File (s)')
            plt.title('Individual Detection Events (size = click count)')
            plt.grid(True, alpha=0.3)

            detail_path = output_path / "detection_events.png"
            plt.savefig(detail_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Detailed plot saved to: {detail_path}")

    def export_summary(self, output_dir):
        """Export summary statistics to file - ENHANCED VERSION"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_text = self.generate_daily_summary()

        # Save summary text
        summary_file = output_path / "enhanced_daily_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_text)

        # Save detailed CSV with dB values
        if not self.stats_df.empty:
            detailed_file = output_path / "enhanced_daily_stats.csv"
            self.stats_df.to_csv(detailed_file, index=False)

        if not self.detections_df.empty:
            detections_file = output_path / "enhanced_daily_detections.csv"
            self.detections_df.to_csv(detections_file, index=False)

        print(f"Enhanced summary exported to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Daily Porpoise Detection Analysis')
    parser.add_argument('results_dir', help='Directory containing detection results')
    parser.add_argument('--output-dir', '-o', default='./enhanced_daily_analysis',
                        help='Output directory for analysis results')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = DailyAnalyzer()

    try:
        # Load data
        print("Loading detection data...")
        analyzer.load_data(args.results_dir)

        # Generate summary
        print("\nGenerating enhanced summary...")
        summary = analyzer.generate_daily_summary()
        print(summary)

        # Create visualizations
        print("\nCreating enhanced visualizations...")
        analyzer.create_visualizations(args.output_dir)

        # Export summary
        print("\nExporting enhanced summary...")
        analyzer.export_summary(args.output_dir)

        print(f"\nEnhanced analysis complete! Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()