import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import os

# -------------------------
# Folder containing MIT-BIH files
# -------------------------
data_folder = "data/mit-bih-arrhythmia-database-1.0.0"

# Records to process
records = ["100","101","102","103","104","105","106","107","108","109"]

all_beats = []

# Keep only these beat types
valid_labels = ['N', 'V', 'A']

for record in records:

    print(f"Processing Record {record}")

    record_path = os.path.join(data_folder, record)

    # Read ECG
    signal, fields = wfdb.rdsamp(record_path)
    Fs = fields['fs']
    ecg = signal[:,0]

    # Read annotations
    ann = wfdb.rdann(record_path, 'atr')

    # High-pass filter
    b,a = butter(4,0.5/(Fs/2),'high')
    ecg = filtfilt(b,a,ecg)

    # Low-pass filter
    b,a = butter(4,40/(Fs/2),'low')
    ecg = filtfilt(b,a,ecg)

    # Notch filter
    b,a = iirnotch(50/(Fs/2),30)
    ecg = filtfilt(b,a,ecg)

    # Detect R peaks
    peaks,_ = find_peaks(ecg,
                         distance=int(0.6*Fs),
                         prominence=0.5)

    # Match detected peaks to annotations
    for sample,label in zip(ann.sample, ann.symbol):

        if label not in valid_labels:
            continue

        idx = np.argmin(np.abs(peaks-sample))

        if abs(peaks[idx]-sample) > 30:
            continue

        if idx==0 or idx>=len(peaks)-1:
            continue

        rr_prev = (peaks[idx]-peaks[idx-1])/Fs
        rr_next = (peaks[idx+1]-peaks[idx])/Fs

        heart_rate = 60/rr_prev

        all_beats.append([
            record,
            rr_prev,
            rr_next,
            heart_rate,
            label
        ])

# Save dataset
df = pd.DataFrame(
    all_beats,
    columns=[
        "Record",
        "RR_Previous",
        "RR_Next",
        "Heart_Rate",
        "Label"
    ]
)

df.to_csv("heartbeat_dataset.csv", index=False)

print("\nDataset Created Successfully!")
print(df.head())
print("\nTotal Beats:", len(df))