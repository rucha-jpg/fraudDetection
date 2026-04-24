import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from datetime import datetime
from collections import deque

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1224 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2a45;
}

/* Force ALL text in sidebar to be visible */
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #38bdf8 !important;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p,
[data-testid="metric-container"] [data-testid="stMetricLabel"] span {
    color: #94a3b8 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] * {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    color: #e2e8f0 !important;
}

/* ── Alert boxes ── */
.alert-fraud {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.05));
    border: 1px solid rgba(239,68,68,0.5);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #fca5a5 !important;
}
.alert-normal {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(22,163,74,0.05));
    border: 1px solid rgba(34,197,94,0.3);
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    margin: 0.25rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #86efac !important;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1e2a45;
}
.page-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.page-header p {
    color: #94a3b8;
    font-size: 0.85rem;
    margin: 0;
}

/* ── Section titles ── */
.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2a45;
}

/* ── Risk badges ── */
.risk-high   { color: #ef4444 !important; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.risk-medium { color: #f59e0b !important; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.risk-low    { color: #22c55e !important; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #1e2a45);
    border: 1px solid #2d4a6e;
    color: #93c5fd !important;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e3a5f);
    border-color: #3b82f6;
    color: #ffffff !important;
    box-shadow: 0 0 16px rgba(59,130,246,0.3);
}

/* ── Inputs ── */
.stNumberInput input {
    background: #0f172a !important;
    border: 1px solid #1e2a45 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
}
.stNumberInput label,
.stNumberInput label p {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    font-weight: 500;
}

/* ── Captions and small text ── */
.stCaption, .stCaption p,
small, caption {
    color: #64748b !important;
}

/* ── Info boxes ── */
.stInfo, .stInfo p {
    color: #94a3b8 !important;
    background: rgba(56,189,248,0.05) !important;
    border-color: rgba(56,189,248,0.2) !important;
}

/* ── Live dot ── */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 1.4s ease-in-out infinite;
    margin-right: 6px;
}
            
/* ── Equal button heights ── */
section[data-testid="stSidebar"] .stButton > button {
    height: 60px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2a45; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "Time": 0.0, "Amount": 0.0,
        "logs": deque(maxlen=200),
        "initialized": True,
        "live_update": False,
        "latest_alert": None,
        "total_tx": 0,
        "total_fraud": 0,
        "prob_history": deque(maxlen=60),
        "amount_history": deque(maxlen=60),
        "ts_history": deque(maxlen=60),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    for i in range(28):
        key = f"V{i+1}"
        if key not in st.session_state:
            st.session_state[key] = 0.0

init_state()

# ─── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("/Users/rucha/fraudDetection/data/creditcard.csv")

df = load_data()

# ─── API Helpers ─────────────────────────────────────────────────────────────
def call_predict(data: dict):
    try:
        r = requests.post(f"{API_URL}/predict", json=data, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.sidebar.error("❌ API offline — run uvicorn in backend terminal")
        return None
    except requests.exceptions.HTTPError as e:
        st.sidebar.warning(f"API error: {e}")
        return None

def call_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def call_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# ─── Build Transaction Dict ───────────────────────────────────────────────────
def build_tx():
    d = {"Time": st.session_state.Time, "Amount": st.session_state.Amount}
    for i in range(28):
        d[f"V{i+1}"] = st.session_state[f"V{i+1}"]
    return d

def load_sample(sample_row):
    st.session_state.Time   = float(sample_row["Time"])
    st.session_state.Amount = float(sample_row["Amount"])
    for i in range(28):
        st.session_state[f"V{i+1}"] = float(sample_row[f"V{i+1}"])

# ─── Plotly Theme ─────────────────────────────────────────────────────────────
PLOT_BG   = "#0d1224"
PLOT_GRID = "#1e2a45"
FONT_CLR  = "#94a3b8"

def base_layout(**kwargs):
    return dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Space Grotesk", color=FONT_CLR, size=11),
        margin=dict(l=40, r=20, t=40, b=40),
        **kwargs
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Fraud Detector")
    st.markdown("---")

    health = call_health()
    if health and health.get("model_loaded"):
        st.success("● API Online & Model Ready")
    elif health:
        st.warning("⚠️ API Online — Model not loaded")
    else:
        st.error("● API Offline")

    st.markdown("---")
    st.markdown("## Simulation")
    auto_mode = st.toggle("⚡ Live Transaction Stream", value=st.session_state.live_update)
    st.session_state.live_update = auto_mode

    if auto_mode:
        speed = st.slider("Refresh (sec)", 0.5, 3.0, 1.0, 0.5)

    st.markdown("---")
    st.markdown("## Quick Load")
    c1, c2, c3 = st.columns(3)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🎲 Rand", use_container_width=True):
            load_sample(df.sample(1).iloc[0]); st.rerun()
    with c2:
        if st.button("🚨 Fraud", use_container_width=True):
            load_sample(df[df["Class"]==1].sample(1).iloc[0]); st.rerun()
    with c3:
        if st.button("✅ Legit", use_container_width=True):
            load_sample(df[df["Class"]==0].sample(1).iloc[0]); st.rerun()

    st.markdown("---")
    st.markdown("## Model Performance")
    stats_data = call_stats()
    if stats_data:
        st.metric("ROC-AUC",  f"{stats_data['roc_auc']:.4f}")
        st.metric("PR-AUC",   f"{stats_data['pr_auc']:.4f}")
        fraud_rate = stats_data.get("fraud_rate", 0)
        st.metric("Fraud Rate", f"{100*fraud_rate:.3f}%")
    else:
        st.info("Run train.py to see model stats")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
  <div>
    <h1>FraudSentinel</h1>
    <p>Real-time credit card fraud detection · XGBoost + SHAP explainability</p>
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.latest_alert:
    st.markdown(f"""
    <div class="alert-fraud">
      🚨 <strong>FRAUD ALERT</strong> &nbsp;|&nbsp;
      Probability: <strong>{st.session_state.latest_alert:.4f}</strong> &nbsp;|&nbsp;
      {datetime.now().strftime("%H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
fraud_rate_live = (
    round(100 * st.session_state.total_fraud / st.session_state.total_tx, 2)
    if st.session_state.total_tx > 0 else 0.0
)
k1.metric("Total Transactions", f"{st.session_state.total_tx:,}")
k2.metric("Fraud Detected",     f"{st.session_state.total_fraud:,}")
k3.metric("Fraud Rate",         f"{fraud_rate_live:.2f}%")
avg_prob = round(np.mean(list(st.session_state.prob_history)), 4) if st.session_state.prob_history else 0.0
k4.metric("Avg Risk Score",     f"{avg_prob:.4f}")

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 1.6], gap="large")

# ══ LEFT: Transaction Input ═══════════════════════════════════════════════════
with left:
    st.markdown('<p class="section-title">Transaction Details</p>', unsafe_allow_html=True)

    if st.session_state.live_update:
        if np.random.rand() < 0.98:
            sample = df[df["Class"]==0].sample(1).iloc[0]
        else:
            sample = df[df["Class"]==1].sample(1).iloc[0]
        load_sample(sample)

    c1, c2 = st.columns(2)
    with c1:
        Time   = st.number_input("Time (sec)", key="Time", format="%.2f")
    with c2:
        Amount = st.number_input("Amount ($)", key="Amount", format="%.4f")

    with st.expander("🔢 PCA Features V1–V28", expanded=False):
        cols = st.columns(4)
        for i in range(28):
            with cols[i % 4]:
                st.number_input(f"V{i+1}", key=f"V{i+1}", format="%.4f")

    predict_btn = st.button("🔍  Analyze Transaction", use_container_width=True)

    if predict_btn or st.session_state.live_update:
        result = call_predict(build_tx())

        if result:
            st.session_state.total_tx    += 1
            st.session_state.total_fraud += result["prediction"]
            st.session_state.prob_history.append(result["probability"])
            st.session_state.amount_history.append(st.session_state.Amount)
            st.session_state.ts_history.append(datetime.now().strftime("%H:%M:%S"))
            st.session_state.logs.appendleft({
                "fraud":  result["prediction"],
                "prob":   result["probability"],
                "risk":   result["risk_level"],
                "amount": st.session_state.Amount,
                "time":   datetime.now().strftime("%H:%M:%S"),
            })

            if result["prediction"] == 1:
                st.session_state.latest_alert = result["probability"]
            else:
                st.session_state.latest_alert = None

            st.markdown("---")
            st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

            if result["prediction"] == 1:
                st.error("🚨 **FRAUD DETECTED**")
            else:
                st.success("✅ **Legitimate Transaction**")

            prob = result["probability"]
            risk = result["risk_level"]
            risk_class = f"risk-{risk.lower()}"

            c1, c2 = st.columns(2)
            c1.metric("Fraud Probability", f"{prob:.4f}")
            c2.markdown(f"<br><span class='{risk_class}'>● {risk} RISK</span>", unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Risk Score", "font": {"size": 13, "color": FONT_CLR}},
                number={"suffix": "%", "font": {"color": "#38bdf8", "size": 22, "family": "JetBrains Mono"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": FONT_CLR, "tickfont": {"size": 10}},
                    "bar":  {"color": "#ef4444" if prob > 0.7 else "#f59e0b" if prob > 0.3 else "#22c55e"},
                    "bgcolor": "#0d1224",
                    "bordercolor": "#1e2a45",
                    "steps": [
                        {"range": [0, 30],  "color": "rgba(34,197,94,0.08)"},
                        {"range": [30, 70], "color": "rgba(245,158,11,0.08)"},
                        {"range": [70, 100],"color": "rgba(239,68,68,0.08)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": prob * 100},
                },
            ))
            fig_gauge.update_layout(**base_layout(height=200))
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            if result.get("top_features"):
                st.markdown('<p class="section-title">SHAP Feature Impact</p>', unsafe_allow_html=True)

                feats  = [f["feature"]    for f in result["top_features"]][::-1]
                values = [f["shap_value"] for f in result["top_features"]][::-1]
                colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]

                fig_shap = go.Figure(go.Bar(
                    x=values, y=feats,
                    orientation="h",
                    marker=dict(color=colors, opacity=0.85),
                    text=[f"{v:+.4f}" for v in values],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=10, color=FONT_CLR),
                ))
                fig_shap.update_layout(
                    **base_layout(height=220),
                    xaxis=dict(gridcolor=PLOT_GRID, zeroline=True,
                               zerolinecolor="#334155", title="SHAP Value (fraud impact)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
                st.caption("🔴 Red = pushes toward fraud   🟢 Green = pushes toward legitimate")

# ══ RIGHT: Dashboard ══════════════════════════════════════════════════════════
with right:
    st.markdown('<p class="section-title">Live Risk Score Timeline</p>', unsafe_allow_html=True)

    if st.session_state.prob_history:
        probs = list(st.session_state.prob_history)

        fig_line = go.Figure()
        fig_line.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(239,68,68,0.07)", line_width=0)
        fig_line.add_hrect(y0=0.3, y1=0.7, fillcolor="rgba(245,158,11,0.05)", line_width=0)
        fig_line.add_trace(go.Scatter(
            x=list(range(len(probs))), y=probs,
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2),
            marker=dict(
                color=["#ef4444" if p > 0.7 else "#f59e0b" if p > 0.3 else "#22c55e" for p in probs],
                size=6, line=dict(width=1, color="#0d1224")
            ),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.06)",
            hovertemplate="Risk: %{y:.4f}<extra></extra>",
        ))
        fig_line.add_hline(y=0.7, line=dict(dash="dot", color="rgba(239,68,68,0.5)", width=1))
        fig_line.add_hline(y=0.3, line=dict(dash="dot", color="rgba(245,158,11,0.4)", width=1))
        fig_line.update_layout(
            **base_layout(height=220),
            xaxis=dict(gridcolor=PLOT_GRID, title="Transaction Index"),
            yaxis=dict(gridcolor=PLOT_GRID, title="Fraud Probability", range=[0, 1]),
            showlegend=False,
        )
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Run predictions to see live timeline")

    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<p class="section-title">Risk Distribution</p>', unsafe_allow_html=True)
        if st.session_state.prob_history:
            fig_hist = go.Figure(go.Histogram(
                x=list(st.session_state.prob_history),
                nbinsx=20,
                marker=dict(
                    color=list(st.session_state.prob_history),
                    colorscale=[[0,"#22c55e"],[0.3,"#f59e0b"],[0.7,"#ef4444"],[1,"#dc2626"]],
                    line=dict(width=0.5, color="#0d1224"),
                ),
            ))
            fig_hist.update_layout(
                **base_layout(height=200),
                xaxis=dict(gridcolor=PLOT_GRID, title="Prob"),
                yaxis=dict(gridcolor=PLOT_GRID, title="Count"),
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No data yet")

    with ch2:
        st.markdown('<p class="section-title">Fraud vs Normal</p>', unsafe_allow_html=True)
        normal = st.session_state.total_tx - st.session_state.total_fraud
        fraud  = st.session_state.total_fraud
        if st.session_state.total_tx > 0:
            fig_pie = go.Figure(go.Pie(
                labels=["Normal", "Fraud"],
                values=[normal, fraud],
                hole=0.6,
                marker=dict(colors=["#22c55e", "#ef4444"],
                            line=dict(color="#0d1224", width=3)),
                textinfo="label+percent",
                textfont=dict(size=11, family="Space Grotesk"),
            ))
            fig_pie.update_layout(
                **base_layout(height=200),
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{fraud}</b><br>fraud",
                    x=0.5, y=0.5,
                    font=dict(size=14, color="#ef4444", family="JetBrains Mono"),
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No data yet")

    st.markdown('<p class="section-title">Transaction Log</p>', unsafe_allow_html=True)

    if st.session_state.live_update:
        st.markdown('<span class="live-dot"></span><small style="color:#22c55e">LIVE</small>', unsafe_allow_html=True)

    if st.session_state.logs:
        for log in list(st.session_state.logs)[:12]:
            if log["fraud"] == 1:
                st.markdown(
                    f'<div class="alert-fraud">'
                    f'🚨 FRAUD &nbsp;·&nbsp; Prob: <b>{log["prob"]:.4f}</b> &nbsp;·&nbsp; '
                    f'${log["amount"]:.2f} &nbsp;·&nbsp; {log["time"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="alert-normal">'
                    f'✓ Normal &nbsp;·&nbsp; {log["prob"]:.4f} &nbsp;·&nbsp; '
                    f'${log["amount"]:.2f} &nbsp;·&nbsp; {log["time"]}</div>',
                    unsafe_allow_html=True
                )

        if st.button("📥 Export Logs as CSV", use_container_width=True):
            log_df = pd.DataFrame(list(st.session_state.logs))
            st.download_button(
                "⬇️ Download CSV", log_df.to_csv(index=False),
                file_name="fraud_logs.csv", mime="text/csv"
            )
    else:
        st.caption("No transactions yet — click Analyze or enable Live Stream")

# ─── Live Refresh ─────────────────────────────────────────────────────────────
if st.session_state.live_update:
    time.sleep(speed if 'speed' in dir() else 1.0)
    st.rerun()