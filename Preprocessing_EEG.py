#!/usr/bin/env python
# coding: utf-8

# ## PREPROCESSING
# 

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
from sklearn.model_selection import train_test_split
import mne
import scipy.io
from Seizure_times import *  # Ensure this .py file is in the same folder or Python path

# Set paths
edf_folder = "/Users/muneezemalik/Desktop/EEG/Raw_EDF_Files"
save_folder = "/Users/muneezemalik/Desktop/EEG/EEG_processed"

# Create save directory if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Initialize lists
CPS_seizures, elec_seizures, noc_seizures, normals = [], [], [], []
labels = []

for patient in [15, 14, 13, 12, 11, 10]:
    if patient == 10:
        files_list = ["Record1.edf", "Record2.edf"]
    elif patient in (15, 13, 11):
        files_list = ["Record1.edf", "Record2.edf", "Record3.edf", "Record4.edf"]
    elif patient in (14, 12):
        files_list = ["Record1.edf", "Record2.edf", "Record3.edf"]
        
    for file_id, file in enumerate(files_list):
        file_path = os.path.join(edf_folder, f"p{patient}_{file}")
        data = mne.io.read_raw_edf(file_path, preload=True)

        # Channel ordering
        if patient in (15, 14, 13, 12, 11):
            drop_list = ['EEG Cz-Ref', 'EEG Pz-Ref', 'ECG EKG', 'Manual']
            data.drop_channels([ch for ch in drop_list if ch in data.ch_names])
        elif patient == 10:
            reorder_list = ['EEG Fp2-Ref', 'EEG Fp1-Ref', 'EEG F8-Ref', 'EEG F4-Ref', 'EEG Fz-Ref',
                            'EEG F3-Ref', 'EEG F7-Ref', 'EEG A2-Ref', 'EEG T4-Ref', 'EEG C4-Ref',
                            'EEG C3-Ref', 'EEG T3-Ref', 'EEG A1-Ref', 'EEG T6-Ref', 'EEG P4-Ref',
                            'EEG P3-Ref', 'EEG T5-Ref', 'EEG O2-Ref', 'EEG O1-Ref']
            data.reorder_channels(reorder_list)

        raw_data = np.array(data.get_data())
        record_time = datetime.combine(date.today(), data.info['meas_date'].time())
        seizure_list = eval(f'seizures_{patient}')[file_id + 1]

        index_mat = np.array([[0, 0]])
        seizure_record = []

        for sz in seizure_list:
            s_time = datetime.combine(date.today(), time(*sz[:3]))
            seizure_duration = sz[3]
            s_index = int((s_time - record_time).total_seconds() * 500)
            s_index_end = s_index + seizure_duration * 500
            st = raw_data[:, s_index:s_index_end]
            index_mat = np.vstack([index_mat, [s_index, s_index_end]])
            seizure_record = np.concatenate((seizure_record, st), axis=1) if seizure_record != [] else st

        # Remove seizure segments from normal data
        all_sz_indices = np.concatenate([np.arange(start, end) for start, end in index_mat[1:]], axis=0)
        normal_record = np.delete(raw_data, all_sz_indices.astype(int), axis=1)

        for i in range(seizure_duration):
            sz_sample = seizure_record[:, i*500:(i+1)*500]
            if patient == 10:
                elec_seizures.append(sz_sample)
            elif patient == 13 and file_id < 3:
                noc_seizures.append(sz_sample)
            else:
                CPS_seizures.append(sz_sample)
            normals.append(normal_record[:, i*500:(i+1)*500])

# Normalize and stack
def normalize(arr):
    arr = np.array(arr)
    return arr / np.amax(np.abs(arr))

CPS_seizures = normalize(CPS_seizures)
elec_seizures = normalize(elec_seizures)
noc_seizures = normalize(noc_seizures)
normals = normalize(normals)

# Combine and label
x = np.vstack((noc_seizures, elec_seizures, CPS_seizures, normals))
labels.extend([3] * len(noc_seizures))
labels.extend([2] * len(elec_seizures))
labels.extend([1] * len(CPS_seizures))
labels.extend([0] * len(normals))

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.1, random_state=1)

# Save as .npy
np.save(os.path.join(save_folder, "x_train.npy"), x_train)
np.save(os.path.join(save_folder, "x_test.npy"), x_test)
np.save(os.path.join(save_folder, "y_train.npy"), y_train)
np.save(os.path.join(save_folder, "y_test.npy"), y_test)

# Save as .mat
scipy.io.savemat(os.path.join(save_folder, "x_train.mat"), {"x_train": x_train})
scipy.io.savemat(os.path.join(save_folder, "x_test.mat"), {"x_test": x_test})
scipy.io.savemat(os.path.join(save_folder, "y_train.mat"), {"y_train": y_train})
scipy.io.savemat(os.path.join(save_folder, "y_test.mat"), {"y_test": y_test})

print("✅ Preprocessing complete. Data saved in:", save_folder)


# In[ ]:




