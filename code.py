import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import scipy.signal as sign
import soundfile as sf

def lowpass_filter(signal, cutoff, fs, order=5):
    b, a = sign.butter(order, cutoff / (0.5 * fs), btype="low")
    return sign.filtfilt(b, a, signal)

def highpass_filter(signal, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sign.butter(order, normal_cutoff, btype='high')
    return sign.filtfilt(b, a, signal)

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sign.butter(order, [low, high], btype="bandpass")
    return sign.filtfilt(b, a, signal)

# --- Load Audio File ---
fs, signal = wavfile.read("modulated_noisy_audio.wav")  # Fs = sample rate, signal = data

# --- FFT Analysis ---
N = len(signal)
yf = fft(signal)
xf = fftfreq(N, 1/fs)

# Positive frequencies
xf_abs = xf[:N//2]
yf_abs = np.abs(yf[:N//2])

# --- Find Carrier Frequency ---
peak_index = np.argmax(yf)
fc = xf_abs[peak_index]
print(f"Estimated Carrier Frequency: {fc:.2f} Hz")

mask = xf >= 0
t = np.arange(len(signal)) / fs
carrier = np.sin(2 * np.pi * fc * t)

cutoff = 0.4 * np.max(yf_abs)
indices = np.where(yf[mask] > cutoff)[0]
band_start = xf[indices[0]]
band_end = xf[indices[-1]]
signal_initial_filter = bandpass_filter(signal, band_start, band_end, fs=fs)

demodulated = signal * np.sin(2 * np.pi * t * fc)
filtered = lowpass_filter(demodulated, cutoff=5000, fs=fs)
filtered = highpass_filter(filtered, cutoff=200, fs=fs)

filtered_demodulated = signal_initial_filter * np.sin(2 * np.pi * fc * t)
filtered_real = lowpass_filter(filtered_demodulated, cutoff=5000, fs=fs)
filtered_real = highpass_filter(filtered_real, cutoff=200, fs=fs)
filtered_real = filtered_real * np.sin(2 * np.pi * t * 200)

filtered_normal = filtered_real / np.max(np.abs(filtered_real))
filtered_int16 = np.int16(filtered_normal * 32767)

wavfile.write("final_output.wav", fs, filtered_int16)

# === SAVE PLOTS ===
time = np.arange(len(signal)) / fs

# 1. Original Signal
plt.figure(figsize=(10, 4))
plt.plot(time, signal)
plt.title("1. Original Modulated Signal (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("1_original_signal.png")
plt.close()

# 2. FFT Spectrum
plt.figure(figsize=(10, 4))
plt.plot(xf_abs, yf_abs)
plt.axvline(fc, color='red', linestyle='--', label=f'Carrier ~ {fc:.2f} Hz')
plt.title("2. Frequency Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()
plt.savefig("2_fft_spectrum.png")
plt.close()

# 3. Bandpass Filtered Signal
plt.figure(figsize=(10, 4))
plt.plot(time, signal_initial_filter)
plt.title("3. Bandpass Filtered Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("3_bandpass_filtered.png")
plt.close()

# 4. Raw Demodulated Signal
plt.figure(figsize=(10, 4))
plt.plot(time, demodulated)
plt.title("4. Raw Demodulated Signal (product with carrier)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("4_demodulated_signal.png")
plt.close()

# 5. Final Filtered Baseband Signal
plt.figure(figsize=(10, 4))
plt.plot(time, filtered_real)
plt.title("5. Final Filtered Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("5_final_filtered.png")
plt.close()

# 6. Final Normalized Output
plt.figure(figsize=(10, 4))
plt.plot(time, filtered_normal)
plt.title("6. Final Normalized Output")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("6_final_normalized.png")
plt.close()
