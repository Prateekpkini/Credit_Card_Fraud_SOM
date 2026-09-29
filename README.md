# High-Dimensional Credit Card Fraud & Anomaly Detection Using Self-Organizing Maps (SOM)

**Visvesvaraya Technological University (VTU) &bull; 7th Semester B.E. &bull; Machine Learning-2 (ML-2) Lab Mini-Project**

---

## 📌 Project Overview
Credit card fraud detection is one of the most demanding problems in financial machine learning due to two fundamental hurdles:
1. **Extreme Class Imbalance:** Fraudulent transactions typically constitute less than 0.2% of real-world banking volume. Supervised learning models (Logistic Regression, Decision Trees) frequently develop extreme majority-class bias, suffering from high false-negative rates unless heavily re-sampled.
2. **High Dimensionality & Evolving Adversarial Patterns:** Fraudsters constantly alter their behavioral modus operandi ("zero-day fraud"). Supervised models trained strictly on past labeled patterns fail against novel evasion tactics.

This project implements an **unsupervised anomaly detection system** using **Kohonen Self-Organizing Maps (SOM)**. Instead of predicting fraud directly from historical labels, the SOM learns the topological manifold of normal consumer behavior in high-dimensional continuous space ($\mathbb{R}^{30}$). Incoming transactions with high **Quantization Error** (large geometric distance from their nearest prototype node) are flagged as anomalies.

---

## 🏗️ Architecture & Theoretical Foundations

```
   Raw Transaction (x ∈ ℝ³⁰)
   [Time, V1...V28, Amount]
             │
             ▼
   MinMaxScaler Preprocessing  ──► Scales all features to [0, 1]
             │
             ▼
   Kohonen SOM 2D Grid (M × M)  ──► Competitive & Cooperative Learning
             │
             ├──────────────────────────┐
             ▼                          ▼
   Best Matching Unit (BMU)       Unified Distance Matrix (U-Matrix)
    c(x) = argmin ||x - w_i||₂    Visual topological landscape of clusters
             │
             ▼
   Quantization Error (QE)
    QE(x) = ||x - w_BMU(x)||₂
             │
             ▼
   Dynamic Percentile Threshold (τ)
   • QE(x) ≥ τ  ──► 🚨 Flagged as Anomalous (Fraud)
   • QE(x) < τ  ──► 🟢 Classified as Normal
```

### 1. Best Matching Unit (BMU) Calculation
For each input transaction vector $\mathbf{x} \in \mathbb{R}^D$ ($D = 30$), the SOM calculates the Euclidean distance to every prototype weight vector $\mathbf{w}_i$:
$$\mathbf{c}(\mathbf{x}) = \arg\min_i \|\mathbf{x} - \mathbf{w}_i\|_2 = \arg\min_i \sqrt{\sum_{j=1}^{D} (x_j - w_{ij})^2}$$

### 2. Topological Neighborhood Function
Unlike standard winner-take-all competitive networks, Kohonen SOM updates the winner and its topological neighbors on the 2D lattice using a Gaussian neighborhood function:
$$h_{ci}(t) = \exp\left( -\frac{\|\mathbf{r}_c - \mathbf{r}_i\|^2}{2\sigma(t)^2} \right)$$
where $\mathbf{r}_c$ and $\mathbf{r}_i$ are discrete 2D grid coordinates, and the radius decays over iterations:
$$\sigma(t) = \frac{\sigma_0}{1 + \frac{t}{T / 2}}$$

### 3. Weight Update Rule
Neurons in the neighborhood shift their weights towards the input vector:
$$\mathbf{w}_i(t+1) = \mathbf{w}_i(t) + \alpha(t) \cdot h_{ci}(t) \cdot \left[ \mathbf{x}(t) - \mathbf{w}_i(t) \right]$$
where $\alpha(t) = \frac{\alpha_0}{1 + \frac{t}{T / 2}}$ is the decaying learning rate.

### 4. Quantization Error & Anomaly Scoring
The Quantization Error ($QE$) measures how well the network reconstructs the transaction:
$$QE(\mathbf{x}) = \|\mathbf{x} - \mathbf{w}_{\mathbf{c}(\mathbf{x})}\|_2$$
- **Normal Transactions:** Map with low $QE$ to densely packed prototype neurons.
- **Fraud Transactions:** Map with elevated $QE$ because the SOM did not dedicate prototypes to eccentric outliers.
- **Decision Rule:**
$$\hat{y}(\mathbf{x}) = \begin{cases} 1 \text{ (Fraud)}, & \text{if } QE(\mathbf{x}) \ge \tau \\ 0 \text{ (Normal)}, & \text{if } QE(\mathbf{x}) < \tau \end{cases}$$

---

## 📁 Repository Structure
```
Credit_Card_Fraud_SOM/
├── app.py              # Interactive Streamlit Web Dashboard
├── som_model.py        # SOM engine wrapper, MiniSom logic, metrics & math
├── data_loader.py      # Dataset loader, Kaggle schema validator & synthetic generator
├── requirements.txt    # Python package dependencies
├── creditcard.csv      # Dataset (loaded or autonomously synthesized)
└── README.md           # Project documentation and lab viva guide
```

---

## 🚀 Setup & Execution Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment tool (`venv`)

### 1. Clone & Navigate
```bash
git clone https://github.com/Prateekpkini/Credit_Card_Fraud_SOM.git
cd Credit_Card_Fraud_SOM
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 🎯 Dashboard Features
1. **Academic Header & Student Details:** Pre-formatted for VTU 7th semester laboratory demonstration.
2. **Interactive Hyperparameter Tuning:** Adjust SOM Grid Dimensions ($10\times10$ to $20\times20$), Learning Rate ($\alpha_0$), Gaussian Neighborhood Radius ($\sigma_0$), Iterations, and Anomaly Percentile Thresholds.
3. **Unified Distance Matrix (U-Matrix):** Dynamic 2D heatmap showing cluster density and high-distance boundary ridges with ground-truth fraud overlays (Red 'X').
4. **Quantization Error Histogram & Cut-off:** Visualizes the dynamic separation between normal and anomalous transaction densities.
5. **Confusion Matrix & Diagnostic Metrics:** Precision, Recall, F1-Score, and ROC-AUC score based on the unsupervised cutoff.
6. **Top Anomalies Table:** Color-coded ranking of the most anomalous transactions with individual feature component drilldowns.

---

## 🎓 VTU External Viva Voce Quick Prep

| Question | Examiner's Focus | Key Model Answer |
| :--- | :--- | :--- |
| **Q1: Why is MinMax scaling mandatory before training a SOM?** | Metric sensitivity | SOM uses Euclidean distance: $d = \sqrt{\sum (x_j - w_j)^2}$. If features have different scales (e.g., `Time` in thousands vs. PCA components in single units), the larger feature dominates the distance, completely biasing the BMU selection. |
| **Q2: What is the difference between Quantization Error and Topographic Error?** | Diagnostic metrics | **Quantization Error (QE)** measures map resolution (average distance from samples to their BMU). **Topographic Error (TE)** measures topology preservation (percentage of samples whose 1st and 2nd BMUs are not adjacent on the grid). |
| **Q3: Why is SOM better suited than supervised algorithms for credit card fraud?** | Problem alignment | Credit card fraud suffers from severe class imbalance ($< 0.5\%$). Supervised classifiers suffer from majority class bias and cannot detect novel zero-day fraud tactics. SOM is unsupervised; it models the manifold of normal behavior, flagging any significant topological departure as an anomaly. |