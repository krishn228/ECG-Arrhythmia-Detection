import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

# ==================================================
# Read ECG Signal
# ==================================================
record_path = "data/mit-bih-arrhythmia-database-1.0.0/100"

signal, fields = wfdb.rdsamp(record_path)

Fs = fields['fs']
ecg = signal[:, 0]

print("=" * 50)
print("ECG SIGNAL INFORMATION")
print("=" * 50)
print("Sampling Frequency :", Fs, "Hz")
print("Signal Shape       :", signal.shape)

# ==================================================
# High-pass Filter (Remove Baseline Wander)
# ==================================================
cutoff = 0.5

b, a = butter(4, cutoff/(Fs/2), btype='high')

ecg_hp = filtfilt(b, a, ecg)

# ==================================================
# Low-pass Filter (Remove High Frequency Noise)
# ==================================================
cutoff = 40

b, a = butter(4, cutoff/(Fs/2), btype='low')

ecg_lp = filtfilt(b, a, ecg_hp)

# ==================================================
# Notch Filter (Remove 50 Hz Powerline Noise)
# ==================================================
f0 = 50
Q = 30

b, a = iirnotch(f0/(Fs/2), Q)

ecg_filtered = filtfilt(b, a, ecg_lp)

# ==================================================
# Detect R Peaks
# ==================================================
peaks, properties = find_peaks(
    ecg_filtered,
    distance=int(0.6 * Fs),
    prominence=0.5
)

print("\nNumber of R Peaks Detected :", len(peaks))

# ==================================================
# RR Interval Calculation
# ==================================================
rr_intervals = np.diff(peaks) / Fs

print("\nFirst 10 RR Intervals (seconds)")
print(rr_intervals[:10])

# ==================================================
# Heart Rate Calculation
# ==================================================
heart_rate = 60 / rr_intervals

average_hr = np.mean(heart_rate)

print("\nAverage Heart Rate : {:.2f} BPM".format(average_hr))

# ==================================================
# Feature Extraction
# ==================================================
mean_rr = np.mean(rr_intervals)
std_rr = np.std(rr_intervals)
max_rr = np.max(rr_intervals)
min_rr = np.min(rr_intervals)

print("\n" + "=" * 50)
print("FEATURE SUMMARY")
print("=" * 50)

print(f"Mean RR Interval : {mean_rr:.4f} sec")
print(f"STD RR Interval  : {std_rr:.4f} sec")
print(f"Maximum RR       : {max_rr:.4f} sec")
print(f"Minimum RR       : {min_rr:.4f} sec")
print(f"Average HR       : {average_hr:.2f} BPM")

# ==================================================
# Save Features to CSV
# ==================================================
features = pd.DataFrame({
    "Mean_RR": [mean_rr],
    "STD_RR": [std_rr],
    "Max_RR": [max_rr],
    "Min_RR": [min_rr],
    "Mean_Heart_Rate": [average_hr]
})

features.to_csv("features.csv", index=False)

print("\nfeatures.csv saved successfully!")

# ==================================================
# Plot ECG with Detected R Peaks
# ==================================================
plt.figure(figsize=(14,5))

plt.plot(ecg_filtered[:2000], label="Filtered ECG")

visible_peaks = peaks[peaks < 2000]

plt.plot(
    visible_peaks,
    ecg_filtered[visible_peaks],
    'ro',
    markersize=6,
    label="Detected R Peaks"
)

plt.title("Filtered ECG with Detected R Peaks")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True)

plt.show()