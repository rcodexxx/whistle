#!/usr/bin/env python3
"""
用於比較 Python 和 MATLAB 聲學分析腳本輸出的 CSV 檔案。

這個精簡版腳本會：
1. 打印一份只包含核心數據的簡潔報告。
2. 生成兩張獨立、清晰的標準頻譜圖以供比較。
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def read_psd_matrix(filepath: Path) -> dict:
    """
    讀取並解析由分析腳本生成的 PSD 矩陣 CSV 檔案。
    """
    if not filepath.exists():
        raise FileNotFoundError(f"錯誤：找不到檔案 {filepath}")

    print(f"正在讀取檔案: {filepath.name}...")
    try:
        data = np.loadtxt(filepath, delimiter=',')

        # 從矩陣中提取各部分
        freqs = data[0, 1:]
        times = data[1:, 0]
        psd_data = data[1:, 1:]

        return {
            "freqs": freqs.astype(np.float32),
            "times": times.astype(np.float32),
            "psd": psd_data.astype(np.float32),
        }
    except Exception as e:
        raise IOError(f"讀取或解析檔案 {filepath} 時發生錯誤: {e}")


def print_summary_report(py_data: dict, mat_data: dict):
    """
    對數據進行比較，並打印一份只包含最重要指標的簡潔報告。
    """
    print("\n" + "=" * 20 + " 核心數據比較報告 " + "=" * 20)

    # 1. 維度比較
    py_shape = py_data['psd'].shape
    mat_shape = mat_data['psd'].shape
    print(f"資料維度 (時間, 頻率):")
    print(f"  - Python: {py_shape}")
    print(f"  - MATLAB: {mat_shape}")

    if py_shape != mat_shape:
        print("警告：維度不匹配！為進行比較，將裁切至最小共同維度。")
        min_rows = min(py_shape[0], mat_shape[0])
        min_cols = min(py_shape[1], mat_shape[1])
        py_data['psd'] = py_data['psd'][:min_rows, :min_cols]
        mat_data['psd'] = mat_data['psd'][:min_rows, :min_cols]

    # 2. 核心 PSD 數據比較
    print("\n--- PSD 數據差異 (dB) ---")
    psd_diff = np.abs(py_data['psd'] - mat_data['psd'])
    mean_diff = np.mean(psd_diff)
    max_diff = np.max(psd_diff)

    print(f"平均差異值: {mean_diff:.4f} dB")
    print(f"最大差異值: {max_diff:.4f} dB")

    # 3. 定位最大差異點
    if max_diff > 0:
        max_idx = np.unravel_index(np.argmax(psd_diff), psd_diff.shape)
        # 確保索引在裁切後的陣列範圍內
        time_idx = max_idx[0]
        freq_idx = max_idx[1]
        if time_idx < len(py_data['times']) and freq_idx < len(py_data['freqs']):
            time_at_max = py_data['times'][time_idx]
            freq_at_max = py_data['freqs'][freq_idx]
            py_val = py_data['psd'][max_idx]
            mat_val = mat_data['psd'][max_idx]
            print(f"最大差異發生位置:")
            print(f"  - 時間: {time_at_max:.2f}s, 頻率: {freq_at_max:.2f}Hz")
            print(f"  - Python 值: {py_val:.2f} dB vs MATLAB 值: {mat_val:.2f} dB")

    print("=" * 52)


def plot_spectrogram(data: dict, title: str, vmin: float = None, vmax: float = None):
    """
    生成單一、清晰的標準頻譜圖。
    新增 vmin 和 vmax 參數以固定顏色範圍。
    """
    plt.figure(figsize=(14, 7))

    psd_data = data['psd']

    # 如果未提供 vmin 或 vmax，則動態計算；否則使用提供的值。
    # 這是為了讓不同圖表之間有可比較的顏色標尺。
    if vmin is None:
        vmin = np.percentile(psd_data[np.isfinite(psd_data)], 5)
    if vmax is None:
        vmax = np.percentile(psd_data[np.isfinite(psd_data)], 99)

    plt.pcolormesh(data['times'], data['freqs'], psd_data.T, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)

    plt.yscale('log')
    plt.title(title, fontsize=16)
    plt.xlabel("時間 (s)")
    plt.ylabel("頻率 (Hz)")

    cbar = plt.colorbar()
    cbar.set_label("PSD (dB re 1 μPa²/Hz)")

    plt.tight_layout()


def main():
    """主執行函式"""
    parser = argparse.ArgumentParser(description="比較 Python 和 MATLAB 的 PSD 輸出 CSV 檔案 (精簡版)。")
    parser.add_argument("python_csv", type=Path, help="由 Python 腳本生成的 CSV 檔案路徑。")
    parser.add_argument("matlab_csv", type=Path, help="由 MATLAB 腳本生成的 CSV 檔案路徑。")
    parser.add_argument("--plot", "-p", action="store_true", help="生成並顯示視覺化比較圖。")
    # 新增指令行參數以控制顏色範圍
    parser.add_argument("--vmin", type=float, default=None, help="頻譜圖顏色的最小值 (dB)。")
    parser.add_argument("--vmax", type=float, default=None, help="頻譜圖顏色的最大值 (dB)。")
    args = parser.parse_args()

    try:
        # 讀取數據
        python_data = read_psd_matrix(args.python_csv)
        matlab_data = read_psd_matrix(args.matlab_csv)

        # 打印簡潔的比較報告
        print_summary_report(python_data, matlab_data)

        # 如果需要，生成頻譜圖
        if args.plot:
            print("\n正在生成頻譜圖...")
            # --- 設定中文字體，解決亂碼問題 ---
            try:
                plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception as e:
                print(f"警告：設定中文字體失敗，圖表標題可能顯示為亂碼。錯誤: {e}")

            # 將 vmin 和 vmax 傳遞給繪圖函式
            plot_spectrogram(python_data, "Python 輸出頻譜圖", vmin=args.vmin, vmax=args.vmax)
            plot_spectrogram(matlab_data, "MATLAB 輸出頻譜圖", vmin=args.vmin, vmax=args.vmax)

            plt.show()

    except (FileNotFoundError, IOError) as e:
        print(e)
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")


if __name__ == '__main__':
    main()
