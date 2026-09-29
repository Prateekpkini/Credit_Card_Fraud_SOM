"""
========================================================================================
High-Dimensional Credit Card Fraud & Anomaly Detection Using Self-Organizing Maps (SOM)
VTU 7th Semester ML-2 Lab Project
Module: SOM Engine & Anomaly Detection Logic (som_model.py)
========================================================================================

Mathematical & Theoretical Foundations for External Viva:
---------------------------------------------------------
A Kohonen Self-Organizing Map (SOM) is an unsupervised, biologically inspired artificial
neural network that performs a non-linear topological projection from a high-dimensional
feature space R^D (here D = 30 credit card features) onto a low-dimensional discrete
lattice (typically a 2D grid of size M x N).

Key Algorithmic Steps:
----------------------
1. Initialization:
   Neuron weight vectors W = {w_i in R^D | i in (1, ..., M*N)} are initialized using
   Principal Component Analysis (PCA) along the first two principal eigenvectors,
   ensuring deterministic initial ordering and faster convergence.

2. Competitive Process (BMU Selection):
   For an incoming input vector x(t) in R^D, all neurons compete. The winner is the
   Best Matching Unit (BMU), denoted c(x), whose weight vector minimizes the Euclidean distance:
       c(x) = argmin_i || x(t) - w_i(t) ||_2 = argmin_i sqrt( sum_{j=1}^D (x_j(t) - w_{ij}(t))^2 )

3. Cooperative Process (Neighborhood Function):
   Neurons topologically adjacent to the BMU on the 2D grid are updated. The topological
   neighborhood h_{ci}(t) is typically a Gaussian kernel:
       h_{ci}(t) = exp( - || r_c - r_i ||^2 / (2 * sigma(t)^2) )
   where r_c and r_i are 2D discrete coordinate vectors of the BMU and neuron i, and
   sigma(t) is the monotonically decreasing neighborhood radius:
       sigma(t) = sigma_0 / (1 + t / (T / 2))

4. Adaptive Process (Weight Update Rule):
   Neurons in the neighborhood shift their weights toward the input vector x(t):
       w_i(t + 1) = w_i(t) + alpha(t) * h_{ci}(t) * [ x(t) - w_i(t) ]
   where alpha(t) is the decaying learning rate:
       alpha(t) = alpha_0 / (1 + t / (T / 2))

5. Anomaly Detection Principle (Quantization Error - QE):
   The Quantization Error for a given sample x is:
       QE(x) = || x - w_{BMU(x)} ||_2
   - Normal transactions form dense, recurring patterns that pull SOM prototypes closely to them,
     resulting in minimal quantization error (QE -> 0).
   - Fraudulent transactions are rare, eccentric behavioral outliers in R^30. No dedicated
     neuron cluster represents them; thus, their Euclidean distance to the nearest BMU is
     significantly elevated (high QE).
   - An anomaly threshold tau (e.g., 95th percentile of QE) partitions the space:
       y_pred(x) = 1 (Fraud) if QE(x) >= tau else 0 (Normal).
"""

import numpy as np
from minisom import MiniSom
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support
)
from typing import Tuple, Dict, Any, Optional


class CreditCardSOM:
    """
    Self-Organizing Map wrapper tailored for Credit Card Fraud & Anomaly Detection.
    Encapsulates training, U-matrix generation, quantization error calculation,
    dynamic thresholding, and diagnostic evaluation.
    """

    def __init__(
        self,
        grid_x: int = 15,
        grid_y: int = 15,
        input_len: int = 30,
        sigma: float = 1.5,
        learning_rate: float = 0.5,
        random_seed: int = 42
    ):
        """
        Initialize the SOM network topology.

        Parameters:
        -----------
        grid_x : int
            Horizontal dimension of the 2D Kohonen grid.
        grid_y : int
            Vertical dimension of the 2D Kohonen grid.
        input_len : int
            Dimensionality of input vectors (30 for Kaggle Credit Card dataset).
        sigma : float
            Initial spread/radius of the Gaussian neighborhood function (sigma_0).
        learning_rate : float
            Initial learning rate (alpha_0).
        random_seed : int
            Seed for reproducible random state.
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        # Instantiate MiniSom engine
        # neighborhood_function='gaussian', topology='rectangular'
        self.som = MiniSom(
            x=grid_x,
            y=grid_y,
            input_len=input_len,
            sigma=sigma,
            learning_rate=learning_rate,
            neighborhood_function="gaussian",
            random_seed=random_seed
        )

        self.is_trained = False
        self.quantization_errors = None
        self.threshold = None

    def train(
        self,
        X_scaled: np.ndarray,
        num_iteration: int = 5000,
        init_method: str = "pca",
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Train the SOM network on normalized feature matrix X_scaled.

        Parameters:
        -----------
        X_scaled : np.ndarray
            MinMax-scaled input matrix of shape (N, input_len).
        num_iteration : int
            Total training iterations / epochs.
        init_method : str
            Weight initialization strategy: 'pca' (linear subspace) or 'random'.
        verbose : bool
            Whether to display training logs.

        Returns:
        --------
        Dict[str, float]:
            Dictionary containing global quantization error and topographic error.
        """
        if init_method.lower() == "pca":
            # PCA initialization spreads weights along principal axes of data variation
            self.som.pca_weights_init(X_scaled)
        else:
            self.som.random_weights_init(X_scaled)

        # Train using random sample sampling per iteration
        self.som.train_random(
            data=X_scaled,
            num_iteration=num_iteration,
            verbose=verbose
        )

        self.is_trained = True

        # Compute global validation metrics on trained map
        # Mean Quantization Error: average distance between data points and their BMU
        mean_qe = self.som.quantization_error(X_scaled)
        
        # Topographic Error: proportion of samples for which the 1st and 2nd BMUs are not adjacent
        # (measures topology preservation quality; ideally close to 0)
        topo_err = self.som.topographic_error(X_scaled)

        # Calculate sample-wise Quantization Error for all transactions
        self.quantization_errors = self.calculate_sample_quantization_errors(X_scaled)

        return {
            "mean_quantization_error": float(mean_qe),
            "topographic_error": float(topo_err)
        }

    def calculate_sample_quantization_errors(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Computes the Euclidean Quantization Error for each individual transaction:
            QE_k = || x_k - w_{BMU(x_k)} ||_2

        Parameters:
        -----------
        X_scaled : np.ndarray
            Normalized transactions array of shape (N, D).

        Returns:
        --------
        np.ndarray:
            1D array of shape (N,) containing quantization error for each sample.
        """
        errors = np.zeros(len(X_scaled), dtype=float)
        weights = self.som.get_weights()  # Shape: (grid_x, grid_y, input_len)

        for i, x in enumerate(X_scaled):
            # Locate BMU 2D grid coordinate (row, col)
            bmu_coord = self.som.winner(x)
            # Retrieve BMU weight vector in high-dimensional space
            bmu_weight = weights[bmu_coord[0], bmu_coord[1]]
            # Euclidean distance in R^D
            errors[i] = np.linalg.norm(x - bmu_weight)

        return errors

    def predict_anomalies(
        self,
        X_scaled: np.ndarray,
        percentile_threshold: float = 95.0
    ) -> Tuple[np.ndarray, float]:
        """
        Performs unsupervised anomaly detection using dynamic percentile cut-off.

        Parameters:
        -----------
        X_scaled : np.ndarray
            Feature matrix (N, D).
        percentile_threshold : float
            Cut-off percentile (e.g., 95.0 flags the top 5% highest QE transactions).

        Returns:
        --------
        Tuple[np.ndarray, float]:
            - Binary prediction flags: 1 for Anomaly (Fraud), 0 for Normal.
            - Computed threshold cut-off value (tau).
        """
        if self.quantization_errors is None:
            self.quantization_errors = self.calculate_sample_quantization_errors(X_scaled)

        # Determine dynamic decision boundary
        tau = np.percentile(self.quantization_errors, percentile_threshold)
        self.threshold = float(tau)

        # Unsupervised classification rule: QE >= tau -> Fraud
        y_pred = (self.quantization_errors >= tau).astype(int)
        return y_pred, self.threshold

    def get_u_matrix(self) -> np.ndarray:
        """
        Computes the Unified Distance Matrix (U-Matrix).
        Each element in the U-Matrix represents the average Euclidean distance
        between the neuron's weight vector and its 4-connected (or 8-connected) neighbors.

        Returns:
        --------
        np.ndarray:
            2D array of shape (grid_x, grid_y) representing the normalized distance map [0, 1].
        """
        return self.som.distance_map()

    def map_samples_to_bmu(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Maps every sample in X_scaled to its BMU coordinates (bmu_x, bmu_y).

        Returns:
        --------
        np.ndarray:
            Array of shape (N, 2) storing integer grid coordinates.
        """
        coords = np.zeros((len(X_scaled), 2), dtype=int)
        for i, x in enumerate(X_scaled):
            winner = self.som.winner(x)
            coords[i] = winner
        return coords

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Computes diagnostic classification and anomaly detection metrics comparing
        unsupervised SOM predictions against ground truth labels.

        Parameters:
        -----------
        y_true : np.ndarray
            Ground truth binary labels (0 = Normal, 1 = Fraud).
        y_pred : np.ndarray
            Predicted binary labels from SOM quantization error threshold.

        Returns:
        --------
        Dict[str, Any]:
            Dictionary containing accuracy, precision, recall, f1, ROC-AUC, and confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        try:
            # Use continuous quantization errors as anomaly score for ROC-AUC
            roc_auc = roc_auc_score(y_true, self.quantization_errors)
        except Exception:
            roc_auc = 0.5

        return {
            "confusion_matrix": cm,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "total_detected_anomalies": int(np.sum(y_pred)),
            "true_frauds": int(np.sum(y_true))
        }
