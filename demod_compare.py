#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal

# 讀取檔案
orig_data, orig_fs = sf.read('data/8423.250606040000.wav', frames=int(60*384000), dtype=np.float32)
demod_data, demod_fs = sf.read('output/8423.250606040000.wav', frames=int(60*75000), dtype=np.float32)

# 處理信號
if orig_data.ndim > 1:
    orig_data = orig_data[:, 0]

if demod_data.ndim > 1:
    complex_signal = demod_data[:, 0] + 1j * demod_data[:, 1]
else:
    complex_signal = demod_data

# 計算PSD
freqs1, psd1 = signal.welch(orig_data, orig_fs, nperseg=8192)
freqs2, psd2 = signal.welch(complex_signal, demod_fs, nperseg=1024)

# 繪圖
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.semilogy(freqs1/1000, psd1, 'b-')
ax1.axvspan(115, 145, alpha=0.3, color='red')
ax1.set_xlim([0, 200])
ax1.set_xlabel('Frequency (kHz)')
ax1.set_ylabel('PSD')
ax1.set_title('Original')
ax1.grid(True, alpha=0.3)

ax2.semilogy(freqs2/1000, np.abs(psd2), 'g-')
ax2.axvspan(-15, 15, alpha=0.3, color='lime')
ax2.set_xlim([-25, 25])
ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('PSD')
ax2.set_title('Demodulated')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('psd_comparison.png', dpi=300)
plt.show()