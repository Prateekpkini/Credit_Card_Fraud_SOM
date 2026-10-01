"""
========================================================================================
High-Dimensional Credit Card Fraud & Anomaly Detection Using Self-Organizing Maps (SOM)
VTU 7th Semester Machine Learning-2 (ML-2) Lab Mini-Project
Main Application UI: app.py (Streamlit Dashboard)
========================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from data_loader import load_credit_card_data, preprocess_data, DATASET_FILENAME
from som_model import CreditCardSOM

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Theme Injection
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud SOM | VTU ML-2 Lab",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic & Professional Presentation
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .academic-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    
    .academic-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        color: #f8fafc;
    }
    
    .academic-subtitle {
        font-size: 1.05rem;
        color: #93c5fd;
        margin-bottom: 14px;
    }
    
    .meta-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    
    .meta-badge {
        background: rgba(255, 255, 255, 0.12);
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    .plot-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

    .viva-box {
        background: #f8fafc;
        border-left: 5px solid #2563eb;
        padding: 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Academic Header & Student Details Placeholder
# -----------------------------------------------------------------------------
st.markdown("""
<div class="academic-header">
    <div class="academic-title">High-Dimensional Credit Card Fraud & Anomaly Detection Using Self-Organizing Maps (SOM)</div>
    <div class="academic-subtitle">Visvesvaraya Technological University (VTU) &bull; 7th Semester B.E. &bull; Machine Learning-2 (ML-2) Lab</div>
    <div class="meta-badge-container">
        <span class="meta-badge"><strong>Candidate:</strong> Prateek Prakash Kini</span>
        <span class="meta-badge"><strong>USN:</strong> 4CB23AI071</span>
        <span class="meta-badge"><strong>Course:</strong> ML-2 Lab Mini Project</span>
        <span class="meta-badge"><strong>Algorithm:</strong> Kohonen Unsupervised SOM</span>
        <span class="meta-badge"><strong>Dimensionality:</strong> 30 Input Features (PCA + Time + Amount)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: SOM Hyperparameters & Controls
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
st.sidebar.title("Configuration & Hyperparameters")
st.sidebar.markdown("Tune the Kohonen SOM topological lattice and anomaly detection thresholds.")

# Dataset Loading Section
st.sidebar.subheader("1. Dataset Settings")
sample_size = st.sidebar.select_slider(
    "Sample Transaction Count (N)",
    options=[1000, 2500, 5000],
    value=5000,
    help="Size of transactions to analyze for snappy lab demonstrations."
)
fraud_rate_pct = st.sidebar.slider(
    "Synthetic Fraud Rate (%)",
    min_value=0.2,
    max_value=2.0,
    value=0.6,
    step=0.1,
    help="Ground truth anomaly injection proportion (typical Kaggle fraud is ~0.17%-0.5%)."
)

# SOM Network Architecture
st.sidebar.subheader("2. SOM Network Architecture")
grid_dim = st.sidebar.slider(
    "Lattice Grid Size (M x M)",
    min_value=10,
    max_value=20,
    value=14,
    step=1,
    help="Dimension of the 2D output Kohonen grid (e.g., 14 gives 196 prototype neurons)."
)

init_method = st.sidebar.selectbox(
    "Weight Initialization",
    options=["PCA (Principal Component Analysis)", "Random Gaussian"],
    index=0,
    help="PCA initialization aligns prototype weights along data variance axes for deterministic, faster convergence."
)
init_key = "pca" if "PCA" in init_method else "random"

# Learning Hyperparameters
st.sidebar.subheader("3. Learning Dynamics")
learning_rate = st.sidebar.slider(
    "Initial Learning Rate (alpha_0)",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Initial magnitude of weight adjustments towards the BMU."
)
sigma = st.sidebar.slider(
    "Initial Neighborhood Radius (sigma_0)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.25,
    help="Standard deviation of the Gaussian neighborhood kernel."
)
iterations = st.sidebar.slider(
    "Training Iterations",
    min_value=1000,
    max_value=10000,
    value=4000,
    step=1000,
    help="Number of competitive training steps."
)

# Anomaly Thresholding
st.sidebar.subheader("4. Anomaly Decision Boundary")
threshold_percentile = st.sidebar.slider(
    "Quantization Error Threshold Cut-off (%)",
    min_value=90.0,
    max_value=99.5,
    value=95.0,
    step=0.5,
    help="Transactions with Quantization Error >= this percentile are flagged as Anomalies (Fraud)."
)

# Button to trigger re-training
train_btn = st.sidebar.button("Train / Re-train SOM Model", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Data Ingestion & Caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset(n_samples: int, fraud_pct: float):
    # Check if dataset exists or generate
    df, is_synthetic = load_credit_card_data(
        n_synthetic_samples=n_samples,
        fraud_ratio=fraud_pct / 100.0,
        random_state=42,
        auto_save_synthetic=True
    )
    # If loaded real dataset is larger, slice to n_samples for speedy interactive training
    if len(df) > n_samples:
        df = df.iloc[:n_samples].copy()
    X_scaled, y, scaler, feature_names = preprocess_data(df)
    return df, X_scaled, y, is_synthetic, feature_names


df, X_scaled, y_true, is_synthetic, feature_names = get_dataset(sample_size, fraud_rate_pct)


# -----------------------------------------------------------------------------
# Model Training Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_som_cached(grid_size, lr, sig, n_iter, init_mode, data_shape_token):
    som_model = CreditCardSOM(
        grid_x=grid_size,
        grid_y=grid_size,
        input_len=30,
        sigma=sig,
        learning_rate=lr,
        random_seed=42
    )
    with st.spinner(f"Training Kohonen SOM ({grid_size}x{grid_size} grid, {n_iter} iterations)..."):
        t0 = time.time()
        metrics = som_model.train(
            X_scaled,
            num_iteration=n_iter,
            init_method=init_mode,
            verbose=False
        )
        elapsed = time.time() - t0
    return som_model, metrics, elapsed


# Unique token representing dataset state
data_token = f"{len(df)}_{y_true.sum()}_{sample_size}"

som, train_metrics, train_time = train_som_cached(
    grid_dim,
    learning_rate,
    sigma,
    iterations,
    init_key,
    data_token
)

# Apply dynamic anomaly thresholding
y_pred, cutoff_tau = som.predict_anomalies(X_scaled, percentile_threshold=threshold_percentile)
eval_results = som.evaluate(y_true, y_pred)


# -----------------------------------------------------------------------------
# KPI Metrics Strip
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_tx = len(df)
n_frauds = int(np.sum(y_true))
fraud_rate = (n_frauds / total_tx) * 100
detected_anomalies = eval_results["total_detected_anomalies"]
recall_val = eval_results["recall"] * 100
prec_val = eval_results["precision"] * 100

with kpi1:
    st.metric(
        label="Total Transactions",
        value=f"{total_tx:,}",
        delta=f"{'Synthetic' if is_synthetic else 'Kaggle Real'} Data"
    )

with kpi2:
    st.metric(
        label="Ground Truth Fraud Rate",
        value=f"{fraud_rate:.2f}%",
        delta=f"{n_frauds} Frauds"
    )

with kpi3:
    st.metric(
        label=f"QE Cutoff (P{threshold_percentile:g})",
        value=f"{cutoff_tau:.4f}",
        delta=f"{eval_results['tp']} True Positives"
    )

with kpi4:
    st.metric(
        label="Flagged Anomalies",
        value=f"{detected_anomalies:,}",
        delta=f"{eval_results['fp']} False Positives"
    )

with kpi5:
    st.metric(
        label="Fraud Detection Recall",
        value=f"{recall_val:.1f}%",
        delta=f"Precision: {prec_val:.1f}%"
    )

st.write("")


# -----------------------------------------------------------------------------
# Main Visualizations
# -----------------------------------------------------------------------------
tab_visuals, tab_data, tab_viva = st.tabs([
    "📊 Visual Analytics & SOM Topography",
    "🔍 Top Anomalies & Data Inspection",
    "🎓 Academic Theory & External Viva Prep"
])

with tab_visuals:
    col_vis1, col_vis2 = st.columns([1.1, 0.9])

    # Visual 1: SOM U-Matrix with Overlaid Normal vs Fraud Markers
    with col_vis1:
        st.markdown("### Visual 1: Unified Distance Matrix (U-Matrix)")
        st.caption("Visualizes topological distances between adjacent prototype neurons. Darker blues indicate dense cluster centers, while bright yellow/white ridges indicate cluster boundaries and outlier zones.")

        fig_umat, ax_umat = plt.subplots(figsize=(7.5, 6.2), dpi=130)
        
        # Calculate U-Matrix
        u_matrix = som.get_u_matrix()
        
        # Plot distance heatmap
        cax = ax_umat.pcolor(u_matrix.T, cmap="viridis", alpha=0.9)
        cbar = plt.colorbar(cax, ax=ax_umat, fraction=0.046, pad=0.04)
        cbar.set_label("Neuron Weight Distance to Neighbors (U-Matrix)", fontsize=9)

        # Map samples to BMUs
        bmu_coords = som.map_samples_to_bmu(X_scaled)

        # Plot Normal samples (subsampled to keep plot crisp and avoid clutter)
        normal_idx = np.where(y_true == 0)[0]
        # Display up to 300 normal samples for visual clarity
        sub_normal = np.random.RandomState(42).choice(normal_idx, size=min(300, len(normal_idx)), replace=False)
        
        # Plot Normal transactions as green circles with slight jitter for visual clarity
        rng_jitter = np.random.RandomState(42)
        ax_umat.scatter(
            bmu_coords[sub_normal, 0] + 0.5 + rng_jitter.uniform(-0.25, 0.25, size=len(sub_normal)),
            bmu_coords[sub_normal, 1] + 0.5 + rng_jitter.uniform(-0.25, 0.25, size=len(sub_normal)),
            s=25,
            c="#22c55e",
            marker="o",
            alpha=0.6,
            edgecolors="none",
            label=f"Normal Tx (subset of {len(normal_idx)})"
        )

        # Plot Fraud transactions as bold red 'X' markers
        fraud_idx = np.where(y_true == 1)[0]
        if len(fraud_idx) > 0:
            ax_umat.scatter(
                bmu_coords[fraud_idx, 0] + 0.5 + rng_jitter.uniform(-0.15, 0.15, size=len(fraud_idx)),
                bmu_coords[fraud_idx, 1] + 0.5 + rng_jitter.uniform(-0.15, 0.15, size=len(fraud_idx)),
                s=95,
                c="#ef4444",
                marker="X",
                linewidths=2.0,
                alpha=0.95,
                label=f"Ground-Truth Fraud (N={len(fraud_idx)})"
            )

        ax_umat.set_xlim(0, grid_dim)
        ax_umat.set_ylim(0, grid_dim)
        ax_umat.set_xticks(range(0, grid_dim + 1, max(1, grid_dim // 5)))
        ax_umat.set_yticks(range(0, grid_dim + 1, max(1, grid_dim // 5)))
        ax_umat.set_xlabel("Kohonen Grid X-Coordinate", fontsize=10)
        ax_umat.set_ylabel("Kohonen Grid Y-Coordinate", fontsize=10)
        ax_umat.set_title(f"SOM Distance Map ({grid_dim}x{grid_dim} Lattice)", fontsize=11, fontweight="bold")
        ax_umat.legend(loc="upper right", framealpha=0.85, fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_umat)
        plt.close(fig_umat)

        st.info("💡 **Examiner Demonstration Tip:** Point out to the examiner how the Red 'X' fraudulent transactions map to peripheral boundary nodes or high-distance ridges (lighter shades), proving that the SOM isolates anomalous behavioral outliers from dense normal clusters.")

    # Visual 2: Quantization Error Distribution & Dynamic Threshold Cutoff
    with col_vis2:
        st.markdown("### Visual 2: Quantization Error (QE) Distribution")
        st.caption("Distribution of Euclidean reconstruction distances ||x - w_BMU||. The dashed red line marks the dynamic anomaly decision threshold.")

        fig_dist, ax_dist = plt.subplots(figsize=(6.8, 4.2), dpi=130)
        qe = som.quantization_errors

        # Plot histogram + KDE
        sns.histplot(
            qe,
            bins=45,
            kde=True,
            color="#3b82f6",
            ax=ax_dist,
            stat="density",
            alpha=0.55
        )

        # Plot Threshold Line
        ax_dist.axvline(
            cutoff_tau,
            color="#dc2626",
            linestyle="--",
            linewidth=2.2,
            label=f"Threshold (P{threshold_percentile:g} = {cutoff_tau:.3f})"
        )

        # Shaded anomaly region
        max_qe = qe.max()
        ax_dist.axvspan(cutoff_tau, max_qe * 1.05, color="#fee2e2", alpha=0.5, label="Flagged Anomaly Zone")

        ax_dist.set_title("Reconstruction Quantization Error Frequency", fontsize=11, fontweight="bold")
        ax_dist.set_xlabel("Quantization Error: || x - w_BMU ||_2", fontsize=10)
        ax_dist.set_ylabel("Probability Density", fontsize=10)
        ax_dist.legend(loc="upper right", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        # Sub-visual: Confusion Matrix
        st.markdown("### Diagnostic Performance Matrix")
        cm_cols = st.columns(2)
        with cm_cols[0]:
            fig_cm, ax_cm = plt.subplots(figsize=(3.8, 3.0), dpi=120)
            sns.heatmap(
                eval_results["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Pred Normal", "Pred Fraud"],
                yticklabels=["True Normal", "True Fraud"],
                ax=ax_cm,
                annot_kws={"size": 10, "weight": "bold"}
            )
            ax_cm.set_title("Confusion Matrix", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with cm_cols[1]:
            st.markdown(f"""
            **Classification Breakdown:**
            - **True Positives (TP):** `{eval_results['tp']}`
            - **False Positives (FP):** `{eval_results['fp']}`
            - **True Negatives (TN):** `{eval_results['tn']}`
            - **False Negatives (FN):** `{eval_results['fn']}`
            - **Precision:** `{eval_results['precision']:.3f}`
            - **Recall (Sensitivity):** `{eval_results['recall']:.3f}`
            - **F1-Score:** `{eval_results['f1_score']:.3f}`
            - **ROC-AUC Score:** `{eval_results['roc_auc']:.3f}`
            """)


# -----------------------------------------------------------------------------
# Tab 2: Top Anomalous Transactions Data Table
# -----------------------------------------------------------------------------
with tab_data:
    st.markdown("### Top Anomalous Transactions Ranked by Quantization Error")
    st.caption("Transactions sorted in descending order of Quantization Error. Higher QE indicates higher geometric divergence from learned normal consumer patterns.")

    # Create inspection dataframe
    df_inspect = df.copy()
    df_inspect["Quantization_Error"] = som.quantization_errors
    df_inspect["Flagged_Anomaly"] = y_pred
    bmu_all = som.map_samples_to_bmu(X_scaled)
    df_inspect["BMU_Node"] = [f"({r},{c})" for r, c in bmu_all]

    # Sort descending
    df_sorted = df_inspect.sort_values(by="Quantization_Error", ascending=False).reset_index(drop=True)

    # Top 10 Anomalies
    top_10 = df_sorted.head(10)

    # Reorder columns for readability
    primary_cols = ["Quantization_Error", "Flagged_Anomaly", "Class", "BMU_Node", "Amount", "Time"]
    pca_cols = [c for c in top_10.columns if c.startswith("V")][:6]  # show first 6 PCA components
    display_cols = primary_cols + pca_cols

    # Format presentation
    try:
        st.dataframe(
            top_10[display_cols].style.format({
                "Quantization_Error": "{:.4f}",
                "Amount": "${:.2f}",
                "Time": "{:.1f}",
                **{c: "{:.3f}" for c in pca_cols}
            }).background_gradient(subset=["Quantization_Error"], cmap="Reds"),
            use_container_width=True
        )
    except Exception:
        st.dataframe(
            top_10[display_cols],
            column_config={
                "Quantization_Error": st.column_config.NumberColumn("Quantization_Error", format="%.4f"),
                "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "Time": st.column_config.NumberColumn("Time", format="%.1f"),
                **{c: st.column_config.NumberColumn(c, format="%.3f") for c in pca_cols}
            },
            use_container_width=True
        )

    st.markdown("#### Full Dataset Anomaly Breakdown Summary")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.write("Ground-Truth Frauds in Top 50 Anomalies:")
        frauds_in_top50 = df_sorted.head(50)["Class"].sum()
        st.success(f"**{frauds_in_top50} out of {n_frauds} total frauds** ({frauds_in_top50 / max(1, n_frauds) * 100:.1f}%) are captured within the top 50 ranked anomalies.")

    with summary_col2:
        st.write("Quantization Error Statistics:")
        st.write(pd.Series(som.quantization_errors).describe().to_frame().T)


# -----------------------------------------------------------------------------
# Tab 3: Academic Theory & VTU External Viva Prep
# -----------------------------------------------------------------------------
with tab_viva:
    st.markdown("## 🎓 Academic Viva Voce Guide & Theoretical Foundations")
    st.caption("Comprehensive notes, mathematical equations, and prepared answers for the VTU ML-2 external laboratory examination.")

    st.markdown("""
    <div class="viva-box">
        <h4>1. Mathematical Formulation of the Best Matching Unit (BMU)</h4>
        <p><strong>Question:</strong> How does the SOM identify the winning neuron for an input transaction vector?</p>
        <p><strong>Answer:</strong> Given an input transaction vector <code>x = [x_1, x_2, ..., x_D]^T &isin; R^D</code> (where D = 30), the algorithm compares <code>x</code> against the weight vector <code>w_i &isin; R^D</code> of every neuron <code>i</code> on the 2D grid. The winner is the neuron with minimum Euclidean distance:</p>
        <code>c(x) = argmin_i || x - w_i ||_2 = argmin_i &radic;(&sum;_{j=1}^{D} (x_j - w_{ij})^2)</code>
    </div>
    
    <div class="viva-box">
        <h4>2. Topological Neighborhood and Weight Update Rule</h4>
        <p><strong>Question:</strong> Explain how neurons learn and why neighboring neurons update together.</p>
        <p><strong>Answer:</strong> Unlike standard competitive learning where only the winner updates, Kohonen SOM preserves topology by updating the BMU and its spatial neighbors. The update formula at iteration <code>t</code> is:</p>
        <code>w_i(t + 1) = w_i(t) + &alpha;(t) &middot; h_{ci}(t) &middot; [x(t) - w_i(t)]</code>
        <p>Where:</p>
        <ul>
            <li><code>&alpha;(t)</code> is the decaying learning rate: <code>&alpha;(t) = &alpha;_0 / (1 + t / (T / 2))</code></li>
            <li><code>h_{ci}(t)</code> is the Gaussian neighborhood kernel: <code>h_{ci}(t) = exp( - || r_c - r_i ||^2 / (2 &sigma;(t)^2) )</code></li>
            <li><code>r_c</code> and <code>r_i</code> are the discrete 2D grid coordinate vectors of the BMU and neuron <code>i</code>.</li>
            <li><code>&sigma;(t)</code> is the shrinking neighborhood radius over iterations.</li>
        </ul>
    </div>

    <div class="viva-box">
        <h4>3. Why are SOMs Superior for High-Dimensional Unsupervised Anomaly Detection?</h4>
        <p><strong>Question:</strong> Why use an unsupervised SOM instead of standard supervised classifiers (e.g., Logistic Regression, Random Forest)?</p>
        <p><strong>Answer:</strong></p>
        <ol>
            <li><strong>Severe Class Imbalance:</strong> In real banking systems, frauds represent &lt; 0.2% of transactions. Supervised classifiers suffer from extreme majority class bias. SOM is completely unsupervised and models the normal manifold directly.</li>
            <li><strong>Zero-Day Novel Fraud Detection:</strong> Supervised models only detect historical fraud patterns. SOM detects any transaction that deviates from the learned normal topology (high Quantization Error), naturally detecting zero-day fraud tactics.</li>
            <li><strong>Dimensionality Reduction & Interpretability:</strong> Maps 30 continuous PCA features to an intuitive 2D U-Matrix, enabling visual auditability for regulatory compliance.</li>
        </ol>
    </div>

    <div class="viva-box">
        <h4>4. Quantization Error vs Topographic Error</h4>
        <p><strong>Question:</strong> What are the two core diagnostic metrics of a trained SOM?</p>
        <p><strong>Answer:</strong></p>
        <ul>
            <li><strong>Quantization Error (QE):</strong> Average distance between each data vector and its BMU prototype: <code>QE = (1/N) &sum; || x_k - w_{BMU(x_k)} ||</code>. Measures resolution/fitting accuracy.</li>
            <li><strong>Topographic Error (TE):</strong> Proportion of all data vectors for which the 1st and 2nd BMUs are not adjacent on the grid lattice. Measures how faithfully the 2D grid preserves the high-dimensional manifold topology (TE &rarr; 0 is optimal).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    VTU 7th Semester B.E. ML-2 Lab Mini Project &bull; Built with Streamlit, MiniSom, Scikit-learn, and Matplotlib.
</div>
""", unsafe_allow_html=True)
