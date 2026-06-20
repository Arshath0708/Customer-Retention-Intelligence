import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Customer Retention Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load machine learning artifacts
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
threshold = joblib.load('optimal_threshold.pkl')
feature_cols = joblib.load('feature_columns.pkl')

# Inject Custom Futuristic Styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;600&display=swap');

/* Force dark background on the entire app container */
[data-testid="stAppViewContainer"] {
    background-color: #080B11 !important;
    background-image: radial-gradient(circle at 50% 0%, #121829 0%, #080B11 100%) !important;
}

[data-testid="stHeader"] {
    background-color: rgba(8, 11, 17, 0) !important;
}

/* Core Typography & Base Styling */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Glowing Title Layout */
.title-container {
    background: linear-gradient(90deg, #00F0FF 0%, #FF007A 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 2.5rem;
    text-shadow: 0 0 25px rgba(0, 240, 255, 0.25);
    margin-bottom: 2px;
}

.subtitle-container {
    font-size: 0.95rem;
    color: #8A99AD;
    letter-spacing: 0.5px;
    margin-bottom: 25px;
}

/* Glassmorphism Cards */
.cyber-card {
    background: rgba(18, 22, 34, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 240, 255, 0.15);
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 12px rgba(0, 240, 255, 0.05);
    margin-bottom: 20px;
    transition: all 0.3s ease;
}

.cyber-card:hover {
    border-color: rgba(0, 240, 255, 0.35);
    box-shadow: 0 8px 32px 0 rgba(0, 240, 255, 0.1), inset 0 0 12px rgba(0, 240, 255, 0.1);
}

/* Streamlit Native Container Styling Override */
div[data-testid="stVerticalBlockBorder"] {
    background: rgba(18, 22, 34, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 240, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
}

/* Sidebar Custom Tech-Styles */
.sidebar-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem;
    color: #FF007A;
    text-shadow: 0 0 8px rgba(255, 0, 122, 0.3);
    margin-top: 15px;
    margin-bottom: 15px;
    letter-spacing: 1.5px;
    border-bottom: 1px solid rgba(255, 0, 122, 0.2);
    padding-bottom: 5px;
}

.sidebar-metric-box {
    background: rgba(21, 26, 40, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
}

.sidebar-label {
    font-size: 0.72rem;
    color: #8A99AD;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}

.sidebar-value {
    font-family: 'Orbitron', sans-serif;
    color: #00F0FF;
    font-size: 1.05rem;
    font-weight: 700;
    text-shadow: 0 0 8px rgba(0, 240, 255, 0.25);
}

.status-online {
    color: #00FF87;
    text-shadow: 0 0 8px rgba(0, 255, 135, 0.35);
    font-weight: bold;
}

/* Native widget labels */
label[data-testid="stWidgetLabel"] p {
    color: #8A99AD !important;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 0.85rem !important;
}

/* Inputs styling */
input, select, textarea {
    background-color: #121622 !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(0, 240, 255, 0.15) !important;
}

div[data-baseweb="select"] > div {
    background-color: #121622 !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(0, 240, 255, 0.15) !important;
}

div[role="listbox"] {
    background-color: #121622 !important;
    border: 1px solid rgba(0, 240, 255, 0.25) !important;
}

div[role="option"] {
    background-color: #121622 !important;
    color: #E2E8F0 !important;
}

div[role="option"]:hover {
    background-color: #1c2336 !important;
    color: #00F0FF !important;
}

/* Buttons Styling */
div.stButton > button {
    background: linear-gradient(135deg, #00F0FF 0%, #0077FF 100%);
    color: #FFFFFF;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    letter-spacing: 1.5px;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    box-shadow: 0 4px 15px rgba(0, 240, 255, 0.25);
    transition: all 0.3s ease;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #00F0FF 20%, #FF007A 100%);
    box-shadow: 0 4px 20px rgba(255, 0, 122, 0.4);
    transform: translateY(-2px);
    color: white;
}

/* Tabs UI overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(18, 22, 34, 0.3);
    border: 1px solid rgba(0, 240, 255, 0.08) !important;
    border-radius: 8px 8px 0px 0px;
    color: #8A99AD !important;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 1.15rem;
    padding: 12px 24px;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(21, 26, 40, 0.9) !important;
    border: 1px solid rgba(0, 240, 255, 0.35) !important;
    border-bottom: none !important;
    color: #00F0FF !important;
    text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Helper function to render Top Metrics Row
def render_metrics_row():
    metrics_html = """
    <div style="display: flex; gap: 15px; margin-bottom: 25px; flex-wrap: wrap;">
        <div class="cyber-card" style="flex: 1; min-width: 250px; border-top: 3px solid #00F0FF; padding: 15px 20px; margin-bottom: 0;">
            <div class="sidebar-label">INTELLIGENCE MODEL</div>
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.3rem; font-weight: 700; color: #FFF; text-shadow: 0 0 8px rgba(0,240,255,0.2); margin-top: 3px;">RANDOM FOREST</div>
            <div style="font-size: 0.75rem; color: #8A99AD; margin-top: 2px;">300 Estimators | stratify=balanced</div>
        </div>
        <div class="cyber-card" style="flex: 1; min-width: 250px; border-top: 3px solid #FF007A; padding: 15px 20px; margin-bottom: 0;">
            <div class="sidebar-label">OPTIMAL DECISION THRESHOLD</div>
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.3rem; font-weight: 700; color: #FFF; text-shadow: 0 0 8px rgba(255,0,122,0.2); margin-top: 3px;">32.0%</div>
            <div style="font-size: 0.75rem; color: #8A99AD; margin-top: 2px;">Tuned dynamically for cost-efficiency</div>
        </div>
        <div class="cyber-card" style="flex: 1; min-width: 250px; border-top: 3px solid #00FF87; padding: 15px 20px; margin-bottom: 0;">
            <div class="sidebar-label">ESTIMATED VALUE IMPROVEMENT</div>
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.3rem; font-weight: 700; color: #FFF; text-shadow: 0 0 8px rgba(0,255,135,0.2); margin-top: 3px;">+€35,700 / Year</div>
            <div style="font-size: 0.75rem; color: #8A99AD; margin-top: 2px;">Total saved: €396,900 (vs. €361,200 baseline)</div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

# Helper function to render dynamic SVG risk gauge
def render_gauge(risk_proba, threshold_val):
    percent = int(risk_proba * 100)
    thresh_percent = int(threshold_val * 100)
    
    if risk_proba >= threshold_val:
        color = "#FF0055"  # Neon pink/red
        glow_class = "glow-red"
        status_text = "HIGH RISK"
        desc_text = "Outreach Recommended Immediately"
    elif risk_proba >= threshold_val * 0.7:
        color = "#FF8C00"  # Neon orange
        glow_class = "glow-orange"
        status_text = "ELEVATED RISK"
        desc_text = "Soft Engagement Suggested"
    else:
        color = "#00FF87"  # Neon green
        glow_class = "glow-green"
        status_text = "LOW RISK"
        desc_text = "Standard Customer Lifecycle"

    stroke_dashoffset = int(440 - (440 * risk_proba))
    
    gauge_html = f"""
    <div class="cyber-card" style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 100%; min-height: 380px; margin-bottom: 0;">
        <div class="sidebar-label" style="margin-bottom: 20px;">RISK ANALYSIS TELEMETRY</div>
        <div class="gauge-container {glow_class}" style="position: relative; width: 180px; height: 180px; margin: 0 auto; display: flex; justify-content: center; align-items: center;">
            <svg width="180" height="180" viewBox="0 0 180 180" style="position: absolute; top: 0; left: 0;">
                <circle cx="90" cy="90" r="70" fill="none" stroke="rgba(255, 255, 255, 0.04)" stroke-width="10"></circle>
                <circle cx="90" cy="90" r="70" fill="none" stroke="{color}" stroke-width="10"
                        stroke-dasharray="440" stroke-dashoffset="{stroke_dashoffset}" stroke-linecap="round"
                        transform="rotate(-90 90 90)" style="transition: stroke-dashoffset 0.8s ease-out;"></circle>
            </svg>
            <div style="z-index: 10; font-family: 'Rajdhani', sans-serif;">
                <div style="font-family: 'Orbitron', sans-serif; font-size: 2.4rem; font-weight: 800; color: {color}; line-height: 1;">{percent}%</div>
                <div style="font-size: 0.72rem; color: #8A99AD; margin-top: 4px; letter-spacing: 1.5px; text-transform: uppercase;">CHURN RISK</div>
            </div>
        </div>
        <div style="margin-top: 25px;">
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.2rem; font-weight: 700; color: {color}; letter-spacing: 1.5px;">{status_text}</div>
            <div style="font-size: 0.8rem; color: #8A99AD; margin-top: 4px; max-width: 220px; margin-left: auto; margin-right: auto;">{desc_text}</div>
        </div>
    </div>
    <style>
    .glow-green {{ filter: drop-shadow(0 0 12px rgba(0, 255, 135, 0.25)); }}
    .glow-orange {{ filter: drop-shadow(0 0 12px rgba(255, 140, 0, 0.25)); }}
    .glow-red {{ filter: drop-shadow(0 0 12px rgba(255, 0, 85, 0.25)); }}
    </style>
    """
    return gauge_html


# Header Section
st.markdown('<div class="title-container">🏦 CUSTOMER RETENTION INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-container">Predictive risk scoring engine & explainable machine learning interface</div>', unsafe_allow_html=True)

# Sidebar (Control Panel)
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ ENGINE CONTROL</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-metric-box">
        <div class="sidebar-label">ENGINE STATUS</div>
        <div class="sidebar-value status-online">● ONLINE</div>
    </div>
    <div class="sidebar-metric-box">
        <div class="sidebar-label">CORE MODEL</div>
        <div class="sidebar-value">Random Forest</div>
    </div>
    <div class="sidebar-metric-box">
        <div class="sidebar-label">MODEL ROC-AUC</div>
        <div class="sidebar-value">0.866</div>
    </div>
    <div class="sidebar-metric-box">
        <div class="sidebar-label">CROSS-VAL ROC-AUC</div>
        <div class="sidebar-value">0.862 ± 0.003</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header" style="color: #00F0FF; border-bottom-color: rgba(0, 240, 255, 0.2);">📊 BUSINESS METRICS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-metric-box">
        <div class="sidebar-label">CUSTOMER VALUE (LTV)</div>
        <div class="sidebar-value">€5,000</div>
    </div>
    <div class="sidebar-metric-box">
        <div class="sidebar-label">RETENTION OFFER COST</div>
        <div class="sidebar-value">€150</div>
    </div>
    <div class="sidebar-metric-box">
        <div class="sidebar-label">OFFER SUCCESS RATE</div>
        <div class="sidebar-value">30.0%</div>
    </div>
    """, unsafe_allow_html=True)

# Render Global Metrics
render_metrics_row()

# Main Interface Tabs
tab1, tab2 = st.tabs(["⚡ Single Customer Lookup", "📂 Bulk Risk Scoring"])

# ── TAB 1: Single Customer Lookup ────────────────────────────────────────
with tab1:
    col_inputs, col_outputs = st.columns([5, 7])
    
    with col_inputs:
        with st.container(border=True):
            st.markdown('<div class="sidebar-label" style="font-size:0.85rem; color:#00F0FF; margin-bottom:20px; font-family:\'Orbitron\', sans-serif;">CUSTOMER PARAMETERS</div>', unsafe_allow_html=True)
            
            credit_score = st.slider("Credit Score", 350, 850, 650)
            age = st.slider("Age", 18, 92, 40)
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Account Balance (€)", 0.0, 250000.0, 50000.0, step=1000.0)
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            salary = st.number_input("Estimated Salary (€)", 0.0, 200000.0, 60000.0, step=1000.0)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            
            chk_col1, chk_col2 = st.columns(2)
            with chk_col1:
                has_card = st.checkbox("Has Credit Card", value=True)
            with chk_col2:
                is_active = st.checkbox("Is Active Member", value=True)
                
            st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
            assess_btn = st.button("Assess Risk", type="primary")

    with col_outputs:
        if assess_btn:
            # Build prediction instance
            row = pd.DataFrame([{
                'CreditScore': credit_score, 'Age': age, 'Tenure': tenure,
                'Balance': balance, 'NumOfProducts': num_products,
                'HasCrCard': int(has_card), 'IsActiveMember': int(is_active),
                'EstimatedSalary': salary,
                'Geography_Germany': int(geography == 'Germany'),
                'Geography_Spain': int(geography == 'Spain'),
                'Gender_Male': int(gender == 'Male'),
            }])
            row = row.reindex(columns=feature_cols, fill_value=0)
            scaled = scaler.transform(row)
            risk = model.predict_proba(scaled)[0][1]

            # Output Grid
            c_out1, c_out2 = st.columns([5, 7])
            with c_out1:
                st.markdown(render_gauge(risk, threshold), unsafe_allow_html=True)
                
            with c_out2:
                with st.container(border=True):
                    st.markdown('<div class="sidebar-label" style="margin-bottom: 12px; font-family:\'Orbitron\', sans-serif;">EXPLAINABLE AI DIAGNOSTIC</div>', unsafe_allow_html=True)
                    
                    # Compute SHAP Values
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(scaled)
                    
                    # Robust handling of SHAP shape format changes
                    if isinstance(shap_values, list):
                        val_to_plot = shap_values[1][0]
                    else:
                        val_to_plot = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                    
                    # Manual High-Fidelity Bar Plot
                    df_importance = pd.DataFrame({
                        'Feature': feature_cols,
                        'SHAP': val_to_plot
                    })
                    df_importance['Abs_SHAP'] = df_importance['SHAP'].abs()
                    df_importance = df_importance.sort_values('Abs_SHAP', ascending=True)
                    
                    colors = ['#FF007A' if x >= 0 else '#00F0FF' for x in df_importance['SHAP']]
                    
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(6.5, 4.5))
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    
                    bars = ax.barh(
                        df_importance['Feature'],
                        df_importance['SHAP'],
                        color=colors,
                        height=0.6,
                        edgecolor=(1, 1, 1, 0.05),
                        linewidth=0.8
                    )
                    
                    ax.grid(axis='x', linestyle='--', alpha=0.15, color='#8A99AD')
                    ax.set_axisbelow(True)
                    ax.tick_params(colors='#8A99AD', labelsize=8.5)
                    
                    for spine in ['top', 'right', 'left', 'bottom']:
                        ax.spines[spine].set_color((1, 1, 1, 0.08))
                        
                    ax.axvline(0, color='white', linewidth=0.7, alpha=0.4)
                    
                    plt.title("FEATURE RETENTION INFLUENCE", fontsize=10, color="#E2E8F0", family="sans-serif", weight="bold", pad=12)
                    plt.xlabel("SHAP Impact Score (← Retains | Churns →)", fontsize=8, color="#8A99AD")
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    st.caption("Hot-magenta bars indicate features pushing customer toward churn. Cyber-cyan bars show retention drivers.")
        else:
            # Welcome/Awaiting Telemetry Display
            st.markdown("""
            <div class="cyber-card" style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 100%; min-height: 380px; margin-bottom: 0;">
                <div style="font-size: 3.5rem; margin-bottom: 12px; filter: drop-shadow(0 0 15px rgba(0, 240, 255, 0.3));">🔮</div>
                <div style="font-family: 'Orbitron', sans-serif; font-size: 1.25rem; font-weight: 700; color: #00F0FF; letter-spacing: 1px;">AWAITING CUSTOMER PROFILE</div>
                <div style="font-size: 0.82rem; color: #8A99AD; margin-top: 8px; max-width: 320px; line-height: 1.4;">
                    Configure customer attributes on the left and click <b>Assess Risk</b> to generate a real-time risk analysis & explainable AI profile.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: Bulk scoring ────────────────────────────────────────────
with tab2:
    with st.container(border=True):
        st.markdown('<div class="sidebar-label" style="font-size:0.85rem; color:#00F0FF; margin-bottom:15px; font-family:\'Orbitron\', sans-serif;">BULK INTELLIGENCE RETRIEVAL</div>', unsafe_allow_html=True)
        st.write("Upload a customer CSV batch file to run predictive scoring and identify high-risk accounts.")
        
        uploaded = st.file_uploader("Select CSV File", type="csv")
        
        if uploaded:
            st.markdown("---")
            bulk_df = pd.read_csv(uploaded)
            bulk_processed = pd.get_dummies(bulk_df, columns=['Geography', 'Gender'], drop_first=True)
            bulk_processed = bulk_processed.reindex(columns=feature_cols, fill_value=0)
            scaled_bulk = scaler.transform(bulk_processed)
            
            bulk_df['churn_risk'] = model.predict_proba(scaled_bulk)[:, 1]
            bulk_df['flagged'] = bulk_df['churn_risk'] >= threshold

            # Compute stats
            total_flagged = bulk_df['flagged'].sum()
            flagged_pct = total_flagged / len(bulk_df)
            
            # Show summary stats in columns
            c_bulk1, c_bulk2, c_bulk3 = st.columns(3)
            with c_bulk1:
                st.markdown(f"""
                <div class="sidebar-metric-box" style="border-top: 2px solid #00F0FF;">
                    <div class="sidebar-label">ACCOUNTS PROCESSED</div>
                    <div class="sidebar-value" style="color: #FFF;">{len(bulk_df):,}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_bulk2:
                st.markdown(f"""
                <div class="sidebar-metric-box" style="border-top: 2px solid #FF0055;">
                    <div class="sidebar-label">FLAGGED AT-RISK</div>
                    <div class="sidebar-value" style="color: #FF0055;">{total_flagged:,} ({flagged_pct:.1%})</div>
                </div>
                """, unsafe_allow_html=True)
            with c_bulk3:
                # Calculate expected value saved if contacted
                # saved = flagged_tp * 30% * 5000 - flagged * 150
                expected_tp_saved = (bulk_df.loc[bulk_df['flagged'], 'churn_risk'].sum()) * 0.30
                est_saved_val = (expected_tp_saved * 5000) - (total_flagged * 150)
                st.markdown(f"""
                <div class="sidebar-metric-box" style="border-top: 2px solid #00FF87;">
                    <div class="sidebar-label">POTENTIAL VALUE GAINED</div>
                    <div class="sidebar-value" style="color: #00FF87;">€{max(0, est_saved_val):,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            top_risk = bulk_df.sort_values('churn_risk', ascending=False).head(15)
            
            st.markdown(f'<div class="sidebar-header" style="font-size: 1rem; color: #FFF; border-bottom: none; margin-top: 15px;">TOP 15 HIGHEST RISK ACCOUNTS</div>', unsafe_allow_html=True)
            
            # Formatting dataframe for presentation
            formatted_df = top_risk.copy()
            if 'CustomerId' in formatted_df.columns:
                display_cols = ['CustomerId', 'Surname', 'Geography', 'Gender', 'churn_risk', 'flagged']
                display_cols = [c for c in display_cols if c in formatted_df.columns]
                formatted_df = formatted_df[display_cols]
            
            # Display dataframe beautifully
            st.dataframe(
                formatted_df.style.format({'churn_risk': '{:.1%}'})
                                  .background_gradient(subset=['churn_risk'], cmap='PuRd')
            )

            # Download Option
            st.markdown('<div style="margin-top:15px;"></div>', unsafe_allow_html=True)
            st.download_button(
                "📥 DOWNLOAD BULK RISK REPORT",
                bulk_df.to_csv(index=False),
                "churn_risk_report.csv"
            )