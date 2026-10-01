# Credit Card Fraud Detection Using Self-Organizing Maps

An unsupervised anomaly detection application built with Python, Streamlit, and MiniSom. The system uses a Kohonen Self-Organizing Map (SOM) to project high-dimensional credit card transactions onto a 2D topological grid and detects fraudulent activities by evaluating quantization errors against a dynamic threshold.

---

## Features

- Unsupervised topological clustering of high-dimensional transaction data using MiniSom.
- Automated data handler that loads `creditcard.csv` or synthesizes a benchmark dataset if unavailable.
- MinMax normalization across all feature dimensions.
- Sample-wise Quantization Error calculation representing distance to the Best Matching Unit (BMU).
- Interactive Streamlit dashboard with sliders for grid dimensions, learning rate, training iterations, and threshold percentile.
- Visual analytics including SOM distance map (U-Matrix), quantization error distribution histogram, and confusion matrix.
- Tabular data inspection for ranked anomalous transactions.

---

## Project Structure

```text
Credit_Card_Fraud_SOM/
├── app.py              # Streamlit web application and visualization layout
├── som_model.py        # MiniSom wrapper, training loop, and anomaly scoring
├── data_loader.py      # Dataset loader, synthetic generator, and preprocessor
├── creditcard.csv      # Credit card transaction dataset
├── requirements.txt    # Project dependencies
└── README.md           # Documentation and run instructions
```

---

## Installation

1. Clone or navigate to the repository directory:
   ```bash
   cd Credit_Card_Fraud_SOM
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

Once running, access the dashboard at:
`http://localhost:8501`