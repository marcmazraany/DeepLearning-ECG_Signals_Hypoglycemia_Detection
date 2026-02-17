import os
import numpy as np
import pandas as pd

df = pd.read_csv("initial Datset/ECG_10s_2500.csv")

# Columns to remove
cols_to_drop = ["count", "day", "glucose_values"]

# Drop unwanted columns if they exist
df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Prepare folder
output_dir = "split_dataset"
os.makedirs(output_dir, exist_ok=True)

# Define patients in each set
train_patients = [3, 4, 7, 8]
val_patients = [1]
test_patients = [2]

# Split the dataframes
train_df = df_clean[df_clean["patient_id"].isin(train_patients)].reset_index(drop=True)
val_df = df_clean[df_clean["patient_id"].isin(val_patients)].reset_index(drop=True)
test_df = df_clean[df_clean["patient_id"].isin(test_patients)].reset_index(drop=True)

# Save as CSV
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print("CSV files saved in 'split_dataset'.")
print("Train:", train_df.shape, "Positives:", train_df['label'].sum())
print("Val:", val_df.shape, "Positives:", val_df['label'].sum())
print("Test:", test_df.shape, "Positives:", test_df['label'].sum())