import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import os

# Folder containing the MIT-BIH records
data_folder = "data/mit-bih-arrhythmia-database-1.0.0"

# Records to process
records = [
    "100","101","102","103","104",
    "105","106","107","108","109"
]

all_features = []

for record in records:

    print(f"Processing Record {record}...")

    record_path = os.path.join(data_folder, record)

    signal, fields = wfdb.rdsamp(record_path)

    Fs = fields['fs']
    ecg = signal[:,0]

    # ------------------------
    # High-pass filter
    # ------------------------
    b,a = butter(4,0.5/(Fs/2),'high')
    ecg = filtfilt(b,a,ecg)

    # ------------------------
    # Low-pass filter
    # ------------------------
    b,a = butter(4,40/(Fs/2),'low')
    ecg = filtfilt(b,a,ecg)

    # ------------------------
    # Notch filter
    # ------------------------
    b,a = iirnotch(50/(Fs/2),30)
    ecg = filtfilt(b,a,ecg)

    # ------------------------
    # Detect R peaks
    # ------------------------
    peaks,_ = find_peaks(
        ecg,
        distance=int(0.6*Fs),
        prominence=0.5
    )

    if len(peaks) < 2:
        print("Skipping Record",record)
        continue

    rr = np.diff(peaks)/Fs

    hr = 60/rr

    mean_rr = np.mean(rr)
    std_rr = np.std(rr)
    max_rr = np.max(rr)
    min_rr = np.min(rr)
    mean_hr = np.mean(hr)

    all_features.append([
        record,
        mean_rr,
        std_rr,
        max_rr,
        min_rr,
        mean_hr
    ])

# -----------------------------------
# Create DataFrame
# -----------------------------------

df = pd.DataFrame(
    all_features,
    columns=[
        "Record",
        "Mean_RR",
        "STD_RR",
        "Max_RR",
        "Min_RR",
        "Mean_Heart_Rate"
    ]
)

df.to_csv("ecg_features_dataset.csv",index=False)

print("\nDataset Created Successfully!")
print(df)