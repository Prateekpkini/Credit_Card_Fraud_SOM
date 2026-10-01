import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_loader import load_credit_card_data, preprocess_data
from som_model import CreditCardSOM

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Using Self-Organizing Maps",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Data Loading & Model Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_data():
    df, is_synthetic = load_credit_card_data()
    X_scaled, y, scaler, feature_names = preprocess_data(df)
    return df, X_scaled, y, feature_names

@st.cache_resource(show_spinner=False)
def train_som(grid_size: int, learning_rate: float, iterations: int, _X_scaled: np.ndarray):
    som = CreditCardSOM(
        grid_x=grid_size,
        grid_y=grid_size,
        input_len=_X_scaled.shape[1],
        sigma=1.0,
        learning_rate=learning_rate,
        random_seed=42
    )
    som.train(_X_scaled, num_iteration=iterations, init_method="pca")
    return som

df, X_scaled, y, feature_names = get_data()

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Model Hyperparameters")

grid_size = st.sidebar.slider(
    "SOM Grid Size (N x N)",
    min_value=5,
    max_value=25,
    value=10,
    step=1
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

iterations = st.sidebar.slider(
    "Training Iterations",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000
)

st.sidebar.header("Anomaly Detection Threshold")
threshold_percentile = st.sidebar.slider(
    "Threshold Percentile",
    min_value=80.0,
    max_value=99.9,
    value=95.0,
    step=0.5
)

# Train or retrieve cached SOM
with st.spinner("Training Self-Organizing Map..."):
    som = train_som(grid_size, learning_rate, iterations, X_scaled)

# Anomaly Predictions
y_pred, threshold_value = som.predict_anomalies(X_scaled, percentile_threshold=threshold_percentile)
quantization_errors = som.quantization_errors

# -----------------------------------------------------------------------------
# Main Layout
# -----------------------------------------------------------------------------
st.title("Credit Card Fraud Detection Using Self-Organizing Maps")

# KPI Metrics
total_transactions = len(df)
fraud_count = int(y.sum())
fraud_rate = (fraud_count / total_transactions) * 100

col_metric1, col_metric2, col_metric3 = st.columns(3)
with col_metric1:
    st.metric(label="Total Transactions", value=f"{total_transactions:,}")
with col_metric2:
    st.metric(label="Fraud Rate", value=f"{fraud_rate:.2f}%")
with col_metric3:
    st.metric(label="Threshold Cut-off", value=f"{threshold_value:.4f}")

st.markdown("---")

# Navigation Tabs
tab_analytics, tab_data = st.tabs(["Visual Analytics", "Data Inspection"])

# -----------------------------------------------------------------------------
# Tab 1: Visual Analytics
# -----------------------------------------------------------------------------
with tab_analytics:
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.subheader("SOM Distance Map (U-Matrix)")
        u_matrix = som.get_u_matrix()
        
        fig_u, ax_u = plt.subplots(figsize=(7, 5))
        p = ax_u.pcolor(u_matrix.T, cmap="bone_r", alpha=0.85)
        cbar = plt.colorbar(p, ax=ax_u)
        cbar.set_label("Neuron Distance (U-Matrix)")

        rng = np.random.RandomState(42)
        normal_indices = np.where(y == 0)[0]
        fraud_indices = np.where(y == 1)[0]

        # Sample normal transactions to prevent overplotting while preserving distribution
        sample_normal = rng.choice(normal_indices, size=min(250, len(normal_indices)), replace=False)

        for idx in sample_normal:
            bmu = som.som.winner(X_scaled[idx])
            ax_u.plot(
                bmu[0] + 0.5 + (rng.rand() - 0.5) * 0.4,
                bmu[1] + 0.5 + (rng.rand() - 0.5) * 0.4,
                "o",
                markerfacecolor="none",
                markeredgecolor="#2563eb",
                markersize=5,
                alpha=0.6
            )

        for idx in fraud_indices:
            bmu = som.som.winner(X_scaled[idx])
            ax_u.plot(
                bmu[0] + 0.5 + (rng.rand() - 0.5) * 0.4,
                bmu[1] + 0.5 + (rng.rand() - 0.5) * 0.4,
                "x",
                color="#dc2626",
                markersize=9,
                markeredgewidth=2
            )

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markeredgecolor="#2563eb", markerfacecolor="none", markersize=7, label="Normal Transaction"),
            Line2D([0], [0], marker="x", color="#dc2626", markeredgewidth=2, markersize=8, label="Fraud Transaction")
        ]
        ax_u.legend(handles=legend_elements, loc="upper right")
        ax_u.set_xlim([0, grid_size])
        ax_u.set_ylim([0, grid_size])
        ax_u.set_xlabel("Neuron Grid X")
        ax_u.set_ylabel("Neuron Grid Y")
        plt.tight_layout()
        st.pyplot(fig_u)
        plt.close(fig_u)

    with col_plot2:
        st.subheader("Quantization Error Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(7, 5))
        ax_hist.hist(quantization_errors, bins=45, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.7)
        ax_hist.axvline(
            threshold_value,
            color="#dc2626",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold_percentile:.1f}th pct = {threshold_value:.4f})"
        )
        ax_hist.set_xlabel("Quantization Error")
        ax_hist.set_ylabel("Transaction Count")
        ax_hist.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    col_cm, _ = st.columns([1, 1])
    with col_cm:
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
            ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("Actual Label")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

# -----------------------------------------------------------------------------
# Tab 2: Data Inspection
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("Raw Data of Most Anomalous Transactions")

    df_inspect = df.copy()
    df_inspect.insert(0, "Quantization_Error", quantization_errors)
    df_inspect.insert(1, "Flagged_Anomaly", y_pred)

    df_sorted = df_inspect.sort_values(by="Quantization_Error", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_sorted.head(100),
        use_container_width=True
    )
