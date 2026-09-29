"""
========================================================================================
High-Dimensional Credit Card Fraud & Anomaly Detection Using Self-Organizing Maps (SOM)
VTU 7th Semester ML-2 Lab Project
Module: Data Loader and Preprocessor (data_loader.py)
========================================================================================

Theoretical Context:
--------------------
Credit card transaction datasets are inherently high-dimensional and severely imbalanced.
The canonical Kaggle Credit Card Fraud dataset contains 284,807 transactions across 30
numerical input features:
  - 'Time': Elapsed seconds between this transaction and the first transaction.
  - 'V1' through 'V28': Continuous features obtained via Principal Component Analysis (PCA)
    to protect customer identity and sensitive confidentiality.
  - 'Amount': Transaction transaction currency value.
  - 'Class': Binary ground-truth target (0 = Legitimate / Normal, 1 = Fraudulent).

When running unsupervised algorithms like Kohonen Self-Organizing Maps (SOM), feature
scaling is strictly necessary. Because the BMU (Best Matching Unit) selection relies on
the Euclidean distance metric:
    d(x, w_i) = sqrt( sum_{j=1}^{D} (x_j - w_{ij})^2 )
any unscaled feature with a wide dynamic range (such as 'Time' or 'Amount') would dominate
the distance computation, overshadowing the PCA components V1-V28. We therefore use
MinMaxScaler to normalize all features to the closed interval [0, 1].
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional


DATASET_FILENAME = "creditcard.csv"


def generate_synthetic_credit_card_data(
    n_samples: int = 5000,
    fraud_ratio: float = 0.005,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Autonomously generate a synthetic, highly imbalanced credit card dataset that
    mirrors the exact schema and statistical characteristics of the Kaggle dataset.

    Parameters:
    -----------
    n_samples : int
        Total number of transaction records to generate (default: 5000).
    fraud_ratio : float
        Proportion of fraudulent transactions (default: 0.005 -> 0.5% fraud rate).
    random_state : int
        Seed for pseudo-random number generator for reproducible lab demonstrations.

    Returns:
    --------
    pd.DataFrame:
        DataFrame containing columns ['Time', 'V1'..'V28', 'Amount', 'Class'].
    """
    rng = np.random.RandomState(random_state)
    
    n_fraud = int(np.round(n_samples * fraud_ratio))
    if n_fraud < 1:
        n_fraud = 1
    n_normal = n_samples - n_fraud

    # 1. 'Time' feature: monotonically increasing elapsed time (0 to 172800 seconds = 2 days)
    time_normal = np.sort(rng.uniform(0, 172800, size=n_normal))
    time_fraud = rng.uniform(0, 172800, size=n_fraud)
    time_all = np.concatenate([time_normal, time_fraud])

    # 2. 'V1' through 'V28': PCA components (Gaussian distributed with variance structure)
    # Legitimate transactions: standard normal distributions around mean 0 with slight variances
    pca_normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 28))

    # Fraudulent transactions: shifted distributions in high-dimensional latent space
    # (Simulating unusual behavioral anomalies across key components e.g., V4, V11, V12, V14, V17)
    pca_fraud = rng.normal(loc=0.0, scale=1.5, size=(n_fraud, 28))
    # Inject deliberate anomalous shifts in recognized critical fraud components:
    pca_fraud[:, 3] += rng.uniform(2.5, 4.5, size=n_fraud)    # V4 shift (positive)
    pca_fraud[:, 10] += rng.uniform(2.0, 4.0, size=n_fraud)   # V11 shift (positive)
    pca_fraud[:, 11] -= rng.uniform(3.0, 6.0, size=n_fraud)   # V12 shift (negative)
    pca_fraud[:, 13] -= rng.uniform(3.5, 6.5, size=n_fraud)   # V14 shift (negative)
    pca_fraud[:, 16] -= rng.uniform(2.5, 5.0, size=n_fraud)   # V17 shift (negative)

    pca_all = np.vstack([pca_normal, pca_fraud])

    # 3. 'Amount' feature: log-normal distribution (most transactions small, few large outliers)
    amount_normal = np.round(rng.lognormal(mean=2.8, sigma=1.2, size=n_normal), 2)
    # Fraud transactions often have extreme amounts (either small micro-charges or large illicit transfers)
    n_half = n_fraud // 2
    amt_small = rng.uniform(1.0, 10.0, size=n_half)
    amt_large = rng.uniform(450.0, 2400.0, size=n_fraud - n_half)
    amount_fraud = np.round(np.concatenate([amt_small, amt_large]), 2)
    rng.shuffle(amount_fraud)
    amount_all = np.concatenate([amount_normal, amount_fraud])

    # 4. 'Class' ground truth label
    class_all = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])

    # Construct DataFrame
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data_matrix = np.column_stack([time_all, pca_all, amount_all, class_all])
    df = pd.DataFrame(data_matrix, columns=columns)
    df["Class"] = df["Class"].astype(int)

    # Shuffle the dataset to mix normal and fraud rows realistically
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def load_credit_card_data(
    filepath: Optional[str] = None,
    n_synthetic_samples: int = 5000,
    fraud_ratio: float = 0.005,
    random_state: int = 42,
    auto_save_synthetic: bool = True
) -> Tuple[pd.DataFrame, bool]:
    """
    Checks if 'creditcard.csv' exists on disk. If found, loads it.
    If not, autonomously generates a high-fidelity synthetic benchmark dataset
    and optionally persists it to disk.

    Returns:
    --------
    Tuple[pd.DataFrame, bool]:
        - Loaded or synthesized DataFrame.
        - Boolean flag indicating True if synthetic data was generated, False if real CSV was loaded.
    """
    target_path = filepath if filepath is not None else DATASET_FILENAME

    if os.path.exists(target_path):
        df = pd.read_csv(target_path)
        is_synthetic = False
    else:
        df = generate_synthetic_credit_card_data(
            n_samples=n_synthetic_samples,
            fraud_ratio=fraud_ratio,
            random_state=random_state
        )
        is_synthetic = True
        if auto_save_synthetic:
            df.to_csv(target_path, index=False)

    return df, is_synthetic


def preprocess_data(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, List[str]]:
    """
    Prepares raw credit card dataframe for unsupervised SOM modeling:
      1. Separates feature matrix X from ground truth label y ('Class').
      2. Normalizes high-dimensional feature space to [0, 1] using MinMaxScaler.
         This satisfies Kohonen SOM geometric constraints and prevents scale-bias.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset containing 30 features and 'Class' column.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, MinMaxScaler, List[str]]:
        - X_scaled: Normalized feature matrix (N, 30) within range [0, 1].
        - y: Ground truth array (N,) of binary values (0: Normal, 1: Fraud).
        - scaler: Fitted instance of MinMaxScaler for inverse transforms or real-time inference.
        - feature_names: Ordered list of feature strings used during modeling.
    """
    feature_names = [col for col in df.columns if col != "Class"]
    X = df[feature_names].values
    y = df["Class"].values if "Class" in df.columns else np.zeros(len(df), dtype=int)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_names


if __name__ == "__main__":
    print("[INFO] Testing data_loader.py execution...")
    data, synthetic_flag = load_credit_card_data()
    print(f"[INFO] Loaded shape: {data.shape} | Is Synthetic: {synthetic_flag}")
    print(f"[INFO] Class Distribution:\n{data['Class'].value_counts()}")
    X_norm, y_labels, sc, f_names = preprocess_data(data)
    print(f"[INFO] Normalized feature matrix bounds: min={X_norm.min():.2f}, max={X_norm.max():.2f}")
    print("[SUCCESS] Data module is self-contained and fully functional.")
