#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
from matplotlib.gridspec import GridSpec


class ComparisonVisualizer:
    def __init__(self, detection_path: Path):
        """載入檢測比較結果"""
        try:
            if detection_path.is_file():
                # 單一CSV檔案
                self.df = pd.read_csv(detection_path)
                self.source_type = "single_file"
            elif detection_path.is_dir():
                # 目錄中的多個比較CSV檔案
                csv_files = list(detection_path.glob("*_comparison.csv"))
                if not csv_files:
                    raise ValueError(f"目錄中未找到 *_comparison.csv 檔案")

                dfs = []
                for csv_file in csv_files:
                    df_temp = pd.read_csv(csv_file)
                    if not df_temp.empty:
                        dfs.append(df_temp)

                if not dfs:
                    raise ValueError("所有CSV檔案都是空的")

                self.df = pd.concat(dfs, ignore_index=True)
                self.source_type = "directory"
                print(f"載入 {len(csv_files)} 個比較檔案")
            else:
                raise ValueError(f"路徑不存在: {detection_path}")

            self.files = self.df['filename'].unique() if not self.df.empty else []
            self.thresholds = sorted(self.df['threshold_factor'].unique()) if not self.df.empty else []
            print(f"載入檢測結果: {len(self.df)} clicks, {len(self.files)} 檔案, {len(self.thresholds)} 個閾值")

        except Exception as e:
            print(f"無法載入檢測結果: {e}")
            sys.exit(1)

    def plot_threshold_comparison(self, output_path: Path = None):
        """繪製閾值比較圖"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 各閾值檢測數量比較
        ax1 = fig.add_subplot(gs[0, :])
        threshold_counts = self.df.groupby('threshold_factor').size()

        bars = ax1.bar(threshold_counts.index, threshold_counts.values,
                       alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('檢測閾值倍數')
        ax1.set_ylabel('檢測到的Click數量')
        ax1.set_title('不同閾值的檢測效果比較', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 添加數值標籤
        for bar, count in zip(bars, threshold_counts.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + count * 0.01,
                     str(count), ha='center', va='bottom', fontweight='bold')

        # 2. 持續時間分布比較
        ax2 = fig.add_subplot(gs[1, 0])
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.thresholds)))

        for i, threshold in enumerate(self.thresholds):
            threshold_data = self.df[self.df['threshold_factor'] == threshold]
            if not threshold_data.empty:
                ax2.hist(threshold_data['duration_us'], bins=30, alpha=0.6,
                         label=f'閾值 {threshold}', color=colors[i], density=True)

        ax2.axvline(111.6, color='red', linestyle='--', linewidth=2, label='期望值: 111.6μs')
        ax2.set_xlabel('持續時間 (μs)')
        ax2.set_ylabel('密度')
        ax2.set_title('各閾值持續時間分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 幅度分布比較
        ax3 = fig.add_subplot(gs[1, 1])

        for i, threshold in enumerate(self.thresholds):
            threshold_data = self.df[self.df['threshold_factor'] == threshold]
            if not threshold_data.empty:
                ax3.hist(threshold_data['max_amplitude'], bins=30, alpha=0.6,
                         label=f'閾值 {threshold}', color=colors[i], density=True)

        ax3.set_xlabel('最大幅度')
        ax3.set_ylabel('密度')
        ax3.set_title('各閾值幅度分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 檔案檢測統計
        ax4 = fig.add_subplot(gs[2, 0])

        # 計算每個檔案在不同閾值下的檢測數量
        file_threshold_counts = self.df.groupby(['filename', 'threshold_factor']).size().unstack(fill_value=0)

        if len(self.files) <= 20:  # 如果檔案不多，顯示所有檔案
            file_threshold_counts.plot(kind='bar', ax=ax4, color=colors[:len(self.thresholds)])
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        else:  # 如果檔案太多，顯示統計
            ax4.text(0.5, 0.5, f'檔案數量過多 ({len(self.files)} 個)\n參見右側統計',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=12)

        ax4.set_xlabel('檔案')
        ax4.set_ylabel('Click數量')
        ax4.set_title('各檔案檢測結果')
        ax4.legend(title='閾值', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)

        # 5. 統計摘要
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        # 計算統計資訊
        stats_text = f"""閾值比較統計摘要
總檔案數: {len(self.files)}
比較閾值: {self.thresholds}

各閾值檢測結果:"""

        for threshold in self.thresholds:
            threshold_data = self.df[self.df['threshold_factor'] == threshold]
            count = len(threshold_data)
            if count > 0:
                mean_duration = threshold_data['duration_us'].mean()
                mean_amplitude = threshold_data['max_amplitude'].mean()
                stats_text += f"""
  閾值 {threshold}: {count} clicks
    平均持續時間: {mean_duration:.1f}μs
    平均幅度: {mean_amplitude:.4f}"""
            else:
                stats_text += f"""
  閾值 {threshold}: 0 clicks"""

        # 檢測率比較
        if len(self.thresholds) > 1:
            base_threshold = min(self.thresholds)
            base_count = len(self.df[self.df['threshold_factor'] == base_threshold])
            stats_text += f"""

相對檢測率 (以閾值{base_threshold}為基準):"""
            for threshold in self.thresholds:
                threshold_count = len(self.df[self.df['threshold_factor'] == threshold])
                if base_count > 0:
                    ratio = threshold_count / base_count * 100
                    stats_text += f"""
  閾值 {threshold}: {ratio:.1f}%"""

        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.suptitle('江豚Click檢測 - 閾值比較分析', fontsize=16, fontweight='bold')

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"比較圖已儲存: {output_path}")

        plt.show()

    def plot_file_detail(self, filename: str, audio_dir: Path = None, output_path: Path = None):
        """繪製單檔案的閾值比較"""
        file_data = self.df[self.df['filename'] == filename]
        if file_data.empty:
            print(f"檔案 {filename} 無檢測結果")
            return

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 時間軸click分布（不同閾值用不同顏色）
        ax1 = fig.add_subplot(gs[0, :])

        # 如果有音檔路徑，讀取並繪製envelope
        if audio_dir:
            audio_path = audio_dir / filename
            if audio_path.exists():
                try:
                    iq_data_raw, fs = sf.read(audio_path, dtype=np.float32)
                    iq_data = iq_data_raw[:, 0] + 1j * iq_data_raw[:, 1]
                    envelope = np.abs(iq_data)

                    # 降採樣顯示（如果太長）
                    if len(envelope) > 500000:
                        downsample_factor = len(envelope) // 200000
                        envelope = envelope[::downsample_factor]
                        fs = fs / downsample_factor

                    t = np.arange(len(envelope)) / fs
                    ax1.plot(t, envelope, 'b-', alpha=0.6, linewidth=0.5, label='Signal Envelope')
                    ax1.set_yscale('log')

                except Exception as e:
                    print(f"無法載入音檔 {filename}: {e}")

        # 標記不同閾值的clicks
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.thresholds)))
        for i, threshold in enumerate(self.thresholds):
            threshold_clicks = file_data[file_data['threshold_factor'] == threshold]
            for _, click in threshold_clicks.iterrows():
                ax1.axvspan(click['start_time'], click['end_time'],
                            alpha=0.6, color=colors[i],
                            label=f'閾值 {threshold}' if _ == threshold_clicks.index[0] else '')

        ax1.set_xlabel('時間 (s)')
        ax1.set_ylabel('幅度')
        ax1.set_title(f'{filename} - 不同閾值檢測結果比較')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. 檢測數量比較
        ax2 = fig.add_subplot(gs[1, 0])
        threshold_counts = file_data.groupby('threshold_factor').size()

        bars = ax2.bar(threshold_counts.index, threshold_counts.values,
                       color=colors[:len(threshold_counts)], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('閾值倍數')
        ax2.set_ylabel('Click數量')
        ax2.set_title('各閾值檢測數量')
        ax2.grid(True, alpha=0.3)

        # 添加數值標籤
        for bar, count in zip(bars, threshold_counts.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + count * 0.05,
                     str(count), ha='center', va='bottom')

        # 3. 持續時間vs閾值
        ax3 = fig.add_subplot(gs[1, 1])

        threshold_durations = []
        threshold_labels = []
        for threshold in self.thresholds:
            threshold_data = file_data[file_data['threshold_factor'] == threshold]
            if not threshold_data.empty:
                threshold_durations.append(threshold_data['duration_us'].values)
                threshold_labels.append(f'{threshold}')

        if threshold_durations:
            ax3.boxplot(threshold_durations, labels=threshold_labels)
            ax3.axhline(111.6, color='red', linestyle='--', alpha=0.7, label='期望值: 111.6μs')
            ax3.set_xlabel('閾值倍數')
            ax3.set_ylabel('持續時間 (μs)')
            ax3.set_title('持續時間分布比較')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '無檢測結果', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=12)
            ax3.set_title('持續時間分布比較')

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"檔案分析圖已儲存: {output_path}")

        plt.show()

    def export_comparison_summary(self, output_path: Path):
        """匯出比較摘要"""
        # 計算各閾值統計
        summary_data = []

        for threshold in self.thresholds:
            threshold_data = self.df[self.df['threshold_factor'] == threshold]

            summary_data.append({
                'threshold_factor': threshold,
                'total_clicks': len(threshold_data),
                'files_with_clicks': threshold_data['filename'].nunique(),
                'mean_duration_us': threshold_data['duration_us'].mean() if not threshold_data.empty else 0,
                'std_duration_us': threshold_data['duration_us'].std() if not threshold_data.empty else 0,
                'mean_amplitude': threshold_data['max_amplitude'].mean() if not threshold_data.empty else 0,
                'std_amplitude': threshold_data['max_amplitude'].std() if not threshold_data.empty else 0
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        print(f"比較摘要已匯出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='江豚Click檢測比較結果視覺化工具')
    parser.add_argument('detection_path', type=Path, help='檢測結果CSV檔案或包含比較檔案的目錄')

    # 分析模式選擇
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--comparison', action='store_true', help='繪製閾值比較圖')
    mode_group.add_argument('--file', type=str, help='分析特定檔案的閾值比較')
    mode_group.add_argument('--export', action='store_true', help='匯出比較摘要')

    # 額外參數
    parser.add_argument('--audio-dir', type=Path, help='音檔目錄路徑（用於讀取原始音檔）')
    parser.add_argument('--output', '-o', type=Path, help='輸出圖片或檔案路徑')

    args = parser.parse_args()

    if not args.detection_path.exists():
        print(f"檢測結果路徑不存在: {args.detection_path}")
        sys.exit(1)

    # 創建視覺化工具
    visualizer = ComparisonVisualizer(args.detection_path)

    # 執行對應功能
    if args.comparison:
        visualizer.plot_threshold_comparison(args.output)

    elif args.file:
        if args.file not in visualizer.files:
            print(f"檔案 {args.file} 不在檢測結果中")
            print(f"可用檔案: {', '.join(visualizer.files[:10])}{'...' if len(visualizer.files) > 10 else ''}")
            sys.exit(1)
        visualizer.plot_file_detail(args.file, args.audio_dir, args.output)

    elif args.export:
        if not args.output:
            if args.detection_path.is_file():
                output_path = args.detection_path.parent / f"{args.detection_path.stem}_summary.csv"
            else:
                output_path = args.detection_path / "comparison_summary.csv"
        else:
            output_path = args.output
        visualizer.export_comparison_summary(output_path)


if __name__ == '__main__':
    main()