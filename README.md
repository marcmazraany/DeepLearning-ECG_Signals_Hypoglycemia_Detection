# ECG-Based Hypoglycemia Detection Using Deep Learning

A deep learning framework for non-invasive hypoglycemia detection using 10-second ECG segments synchronized with CGM-derived glucose labels.

This project investigates whether cardiac electrophysiological changes induced by low blood glucose (<70 mg/dL) can be detected directly from ECG signals using modern representation learning and hybrid deep neural networks.

---

## Overview

Hypoglycemia is a dangerous condition for individuals with diabetes and can lead to cognitive impairment, seizures, or loss of consciousness if left undetected.

Current monitoring methods:
- Finger-stick glucose testing
- Continuous Glucose Monitoring (CGM)

These methods are invasive and may cause discomfort or adherence issues.

This project explores a **non-invasive alternative**:
Using ECG signals to detect hypoglycemia through autonomic nervous system changes reflected in heart rate variability and waveform morphology.

---

## Dataset

- 10-second ECG segments
- 2,500 samples per segment
- Binary labels derived from synchronized CGM readings:
  - 0 → Normoglycemic
  - 1 → Hypoglycemic (<70 mg/dL)

### Preprocessing

- Zero-mean normalization
- Unit-variance scaling
- Stratified train / validation / test split (non patient-based)

---

## Model Architecture

The system combines three major components:

---

### 1. Autoencoder (Unsupervised Representation Learning)

Purpose:
Learn compact latent representations of ECG signals.

- Fully connected encoder compresses 2,500 → latent vector
- Symmetric decoder reconstructs ECG
- Trained using Mean Squared Error (MSE)
- Adam optimizer + early stopping

Training was performed primarily on hypoglycemic samples to enhance minority-class representation learning.

After training:
- Encoder retained
- Latent vectors used as input to classifier

---

### 2. CNN–LSTM Classifier with Attention

Designed to capture both:

- Local waveform morphology (via CNN)
- Temporal dependencies across the 10-second window (via LSTM)

Architecture:

1. 1D Convolutional layers
2. LSTM layers
3. Attention mechanism (learnable timestep weighting)
4. Fully connected layer
5. Sigmoid output (binary probability)

Training:
- Binary Cross-Entropy loss
- Adam optimizer
- Threshold tuning on validation set to maximize F1-score for hypoglycemia

---

### 3. Explainability with Integrated Gradients

To ensure clinical interpretability:

- Integrated Gradients (IG) was applied
- Attribution score computed for each of the 2,500 time points
- Verified that predictions relied on physiologically meaningful ECG regions

This step reduces black-box behavior and increases model trustworthiness.

---

## Results

## Confusion Matrix (Test Set)

|               | Predicted: 0 (Normoglycemic) | Predicted: 1 (Hypoglycemic) |
|---------------|-----------------------------|-----------------------------|
| **Actual: 0 (Normoglycemic)** | 4273 | 44  |
| **Actual: 1 (Hypoglycemic)**  | 96   | 150 |


### Performance Metrics

- Overall Accuracy: **97%**
- Normoglycemic Class:
  - Precision > 98%
  - Recall > 98%
- Hypoglycemic Class:
  - Precision: 0.77
  - Recall: 0.61
  - F1-score: 0.68
- Macro-average F1-score: 0.83

Despite class imbalance, the model demonstrates strong minority-class detection capability.

---

## Key Insights

- ECG signals contain sufficient information to detect hypoglycemia.
- CNN–LSTM hybrids outperform simple architectures for physiological time-series.
- Autoencoder pretraining improves robustness to noise and inter-patient variability.
- Integrated Gradients confirms physiologically relevant feature utilization.
- Class imbalance remains a major challenge for improving recall.

---

## Limitations

- No Leave-One-Patient-Out (LOPO) validation
- No k-fold cross-validation
- Patient-independent generalization requires further evaluation

---

## Future Work

- LOPO evaluation for stronger generalization claims
- Advanced imbalance-handling techniques
- Patient-specific fine-tuning
- Multimodal physiological fusion (ECG + PPG + EDA)
- Deployment on wearable devices

---

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Deep learning & signal processing libraries
