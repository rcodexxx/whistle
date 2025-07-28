import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.signal import butter, sosfilt, welch, decimate, spectrogram

# --- 步驟 0: 設定圖表標題 (全英文) ---
# 為了確保在任何環境下都能正常執行，所有標題均使用英文。
titles = {
    'main': 'Audio Analysis Report',
    'psd': '1. Power Spectral Density (after High-pass)',
    'spec_orig': '2. Original Signal Spectrogram (after High-pass)',
    'spec_final': '3. Final Baseband Signal Spectrogram (Decimated)',
    'fft_final': '4. Final Baseband Signal Spectrum',
    'freq': 'Frequency (kHz)',
    'power': 'Power/Frequency (dB/Hz)',
    'peak': 'Detected Peak',
    'time': 'Time (s)',
    'amp': 'Amplitude'
}


def analyze_large_wav_segment(filepath, duration_to_load_s=60, target_fs=48000, highpass_cutoff_hz=2000):
    """
    載入大型 WAV 檔的前 N 秒，分析其中心頻率，
    然後將其解調至基頻並降採樣，最後繪製所有分析圖。
    """
    print(f"--- Starting analysis for: {filepath} ---")

    # --- 步驟 1: 讀取 WAV 檔的前 N 秒 ---
    try:
        with wave.open(filepath, 'rb') as wf:
            fs = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames_total = wf.getnframes()
            frames_to_read = min(int(duration_to_load_s * fs), n_frames_total)

            print(f"Original sample rate: {fs} Hz")
            print(f"Total duration: {n_frames_total / fs:.1f} s")
            print(f"Reading first {frames_to_read / fs:.1f} seconds of data...")

            frames_bytes = wf.readframes(frames_to_read)
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            data = np.frombuffer(frames_bytes, dtype=dtype_map.get(sampwidth, np.int16))

            if n_channels > 1:
                data = data[::n_channels]

            data_normalized = data.astype(np.float64) / np.iinfo(data.dtype).max
    except Exception as e:
        print(f"Error processing file: {e}")
        return

    # --- 步驟 2: 高通濾波以利偵測 ---
    print(f"\n--- Applying high-pass filter to remove noise below {highpass_cutoff_hz} Hz... ---")
    nyquist = 0.5 * fs
    sos = butter(5, highpass_cutoff_hz / nyquist, btype='high', output='sos')
    data_for_detection = sosfilt(sos, data_normalized)

    # --- 步驟 3: 分析中心頻率 (使用已濾波的數據) ---
    print("\n--- Analyzing center frequency... ---")
    frequencies, psd = welch(data_for_detection, fs, nperseg=8192)
    peak_index = np.argmax(psd)
    detected_center_freq = frequencies[peak_index]
    print(f"Detected center frequency: {detected_center_freq / 1000:.2f} kHz")

    # --- 步驟 4: 解調與降採樣 (使用原始未濾波的數據) ---
    print("\n--- Demodulating to baseband and decimating... ---")
    t = np.arange(len(data_normalized)) / fs
    carrier_I = np.cos(2 * np.pi * detected_center_freq * t)
    carrier_Q = -np.sin(2 * np.pi * detected_center_freq * t)

    I_signal = data_normalized * carrier_I
    Q_signal = data_normalized * carrier_Q

    decimation_factor = int(fs / target_fs)
    print(f"Decimation factor: {decimation_factor} (from {fs / 1000} kHz -> {target_fs / 1000} kHz)")

    I_final = decimate(I_signal, decimation_factor)
    Q_final = decimate(Q_signal, decimation_factor)

    complex_final_signal = I_final + 1j * Q_final
    fs_new = fs // decimation_factor

    # --- 步驟 5: 整合視覺化 ---
    print("\n--- Plotting results... ---")
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(f'{titles["main"]}: {filepath.split("/")[-1]}', fontsize=16)

    # 圖 1: PSD
    axs[0, 0].semilogy(frequencies / 1000, psd)
    axs[0, 0].set_title(titles['psd'])
    axs[0, 0].set_xlabel(titles['freq'])
    axs[0, 0].set_ylabel(titles['power'])
    axs[0, 0].axvline(x=detected_center_freq / 1000, color='r', linestyle='--',
                      label=f'{titles["peak"]}: {detected_center_freq / 1000:.2f} kHz')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_xlim(left=0)

    # 圖 2: 原始訊號時頻譜圖
    duration_for_spec_plot = 10
    samples_for_spec = int(fs * duration_for_spec_plot)
    f_orig, t_orig, Sxx_orig = spectrogram(data_for_detection[:samples_for_spec], fs, nperseg=2048, noverlap=1024)
    axs[0, 1].pcolormesh(t_orig, f_orig / 1000, 10 * np.log10(Sxx_orig), shading='gouraud', rasterized=True)
    axs[0, 1].set_title(f"{titles['spec_orig']} (first {duration_for_spec_plot} s)")
    axs[0, 1].set_xlabel(titles['time'])
    axs[0, 1].set_ylabel(titles['freq'])
    axs[0, 1].axhline(y=detected_center_freq, color='r', linestyle='--')
    axs[0, 1].set_ylim(max(0, detected_center_freq / 1000 - 25), detected_center_freq / 1000 + 25)

    # 圖 3: 最終基頻訊號時頻譜圖
    f_final, t_final, Sxx_final = spectrogram(complex_final_signal, fs_new, nperseg=256, noverlap=128,
                                              return_onesided=False)
    axs[1, 0].pcolormesh(t_final, np.fft.fftshift(f_final) / 1000, np.fft.fftshift(10 * np.log10(Sxx_final), axes=0),
                         shading='gouraud', rasterized=True)
    axs[1, 0].set_title(titles['spec_final'])
    axs[1, 0].set_xlabel(titles['time'])
    axs[1, 0].set_ylabel(titles['freq'])

    # 圖 4: 最終基頻訊號頻譜
    n_final = len(complex_final_signal)
    fft_final = np.fft.fft(complex_final_signal)
    freq_final = np.fft.fftfreq(n_final, 1 / fs_new)
    axs[1, 1].plot(np.fft.fftshift(freq_final) / 1000, np.fft.fftshift(np.abs(fft_final)))
    axs[1, 1].set_title(titles['fft_final'])
    axs[1, 1].set_xlabel(titles['freq'])
    axs[1, 1].set_ylabel(titles['amp'])
    axs[1, 1].grid(True)

    plt.show()


if __name__ == '__main__':
    # --- 主程式設定 ---
    WAV_FILE_PATH = './data/8423.250606040000.wav'

    analyze_large_wav_segment(
        filepath=WAV_FILE_PATH,
        duration_to_load_s=60,
        target_fs=48000,
        highpass_cutoff_hz=2000
    )
