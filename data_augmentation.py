import os
import numpy as np
import pandas as pd

# Augmentation functions
def add_noise(x, noise_level=0.01):
    return x + np.random.normal(0, noise_level, size=x.shape)

def scale_signal(x, min_scale=0.9, max_scale=1.1):
    return x * np.random.uniform(min_scale, max_scale)

def shift_signal(x, max_shift=20):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift)

def augment_signal(x):
    ops = [add_noise, scale_signal, shift_signal]
    num_ops = np.random.choice([1, 2])
    chosen = np.random.choice(ops, num_ops, replace=False)
    
    x_aug = x.copy()
    for op in chosen:
        x_aug = op(x_aug)
    return x_aug

def augment_minority_dataframe(df, multiplier=3):
    augmented_rows = []
    ecg_cols = [str(i) for i in range(2500)]  # ECG sample columns

    for _, row in df[df["label"] == 1].iterrows():
        x = row[ecg_cols].values.astype(float)
        for _ in range(multiplier):
            x_aug = augment_signal(x)
            new_row = row.copy()
            new_row[ecg_cols] = x_aug
            augmented_rows.append(new_row)

    return pd.DataFrame(augmented_rows)


# Load original split files
split_path = "split_dataset"

train_df = pd.read_csv(os.path.join(split_path, "train.csv"))
val_df   = pd.read_csv(os.path.join(split_path, "val.csv"))
test_df  = pd.read_csv(os.path.join(split_path, "test.csv"))

# Apply augmentation only to the training set
aug_df = augment_minority_dataframe(train_df, multiplier=3)

# Combine original and augmented training samples
train_aug_df = pd.concat([train_df, aug_df], ignore_index=True)

# Prepare output folder
out_path = "split_dataset_augmented"
os.makedirs(out_path, exist_ok=True)

# Save new CSV files
train_aug_df.to_csv(os.path.join(out_path, "train_aug.csv"), index=False)
val_df.to_csv(os.path.join(out_path, "val.csv"), index=False)
test_df.to_csv(os.path.join(out_path, "test.csv"), index=False)

print("Augmented dataset saved in folder:", out_path)
print("Train original:", train_df.shape)
print("Augmented samples added:", aug_df.shape)
print("Train new:", train_aug_df.shape)