"""
Remove outliers from the preprocessed Shakespeare dataset.
For illustration purposes, we filter out tokens with IDs > 60 (rare punctuation).
"""
import os
import numpy as np
import pickle

# load the preprocessed data
train_ids = np.fromfile('train.bin', dtype=np.uint16)
val_ids = np.fromfile('val.bin', dtype=np.uint16)

# load metadata
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)

# simple outlier removal: filter out tokens > 60
# (these would be rare punctuation in the Shakespeare dataset)
print(f"Original train size: {len(train_ids):,}")
print(f"Original val size: {len(val_ids):,}")

train_mask = train_ids <= 60
val_mask = val_ids <= 60

train_filtered = train_ids[train_mask]
val_filtered = val_ids[val_mask]

print(f"Filtered train size: {len(train_filtered):,}")
print(f"Filtered val size: {len(val_filtered):,}")
print(f"Removed {len(train_ids) - len(train_filtered):,} outliers from train")
print(f"Removed {len(val_ids) - len(val_filtered):,} outliers from val")

# save filtered data to new files
train_filtered.tofile('train_filtered.bin')
val_filtered.tofile('val_filtered.bin')

# save metadata to new file
with open('meta_filtered.pkl', 'wb') as f:
    pickle.dump(meta, f)