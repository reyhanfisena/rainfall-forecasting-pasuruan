"""
Aplikasi Streamlit — Prediksi Curah Hujan LSTM-PSO
BMKG Stasiun Geofisika Pasuruan | 2021–2025
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Opsional: import model jika tersedia ──────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Prediksi Curah Hujan — Pasuruan",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Light Theme) ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── FORCE LIGHT BASE ───────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #1c2128 !important;
    }
    .main, .stApp, .stAppViewContainer,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #f6f8fa !important;
    }
    /* Top toolbar */
    [data-testid="stHeader"],
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #d0d7de !important;
    }
    [data-testid="stToolbar"] { color: #57606a !important; }

    /* ── OVERRIDE SEMUA NATIVE WIDGET GELAP ────────────────────── */
    /* Selectbox & Multiselect */
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div:hover {
        background-color: #ffffff !important;
        border-color: #d0d7de !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: #1c2128 !important;
        background-color: transparent !important;
    }
    [data-baseweb="select"] svg { fill: #57606a !important; }

    /* Dropdown popover / menu */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 1px solid #d0d7de !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.10) !important;
    }
    li[role="option"],
    [data-baseweb="option"] {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }
    li[role="option"]:hover,
    [data-baseweb="option"]:hover {
        background-color: #f0f6ff !important;
    }
    li[aria-selected="true"],
    [aria-selected="true"][data-baseweb="option"] {
        background-color: #dbeafe !important;
        color: #1c2128 !important;
    }

    /* Number input & text input */
    input, textarea,
    [data-baseweb="input"] input,
    .stNumberInput input,
    .stTextInput input {
        background-color: #ffffff !important;
        color: #1c2128 !important;
        border-color: #d0d7de !important;
    }
    [data-baseweb="input"],
    [data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border-color: #d0d7de !important;
    }

    /* Labels & text */
    label, .stMarkdown p, .stMarkdown span,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        color: #1c2128 !important;
    }

    /* Caption / small text */
    [data-testid="stCaptionContainer"] p,
    .stCaption { color: #57606a !important; }

    /* Code block */
    code, pre { background-color: #f0f6ff !important; color: #0550ae !important; }

    /* Expander */
    [data-testid="stExpander"] details {
        background-color: #ffffff !important;
        border: 1px solid #d0d7de !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #1c2128 !important;
    }

    /* ── HERO HEADER ────────────────────────────────────────────── */
    .hero-header {
        background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 50%, #f0f9ff 100%);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(2,132,199,0.10) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #1e3a5f !important;
        margin: 0 0 6px 0 !important;
        letter-spacing: -0.5px !important;
    }
    .hero-sub {
        font-size: 14px !important;
        color: #475569 !important;
        margin: 0 !important;
        font-weight: 500 !important;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(2,132,199,0.10);
        border: 1px solid rgba(2,132,199,0.30);
        color: #0369a1 !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── METRIC CARDS ───────────────────────────────────────────── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 12px;
        padding: 20px 22px;
        text-align: center;
        transition: border-color 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-card:hover { border-color: #0969da; }
    .metric-label {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #57606a !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #1c2128 !important;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
    }
    .metric-good  { color: #1a7f37 !important; }
    .metric-ok    { color: #b45309 !important; }
    .metric-model { font-size: 11px; color: #0969da !important; margin-top: 4px; font-weight: 600; }

    /* ── SECTION TITLE ──────────────────────────────────────────── */
    .section-title {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #1c2128 !important;
        margin: 24px 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #d0d7de;
    }

    /* ── INFO / WARNING / RESULT BOXES ─────────────────────────── */
    .info-box {
        background: rgba(9,105,218,0.06);
        border: 1px solid rgba(9,105,218,0.20);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 13px;
        color: #0550ae !important;
    }
    .info-box strong, .info-box b { color: #0550ae !important; }
    .warning-box {
        background: rgba(180,83,9,0.06);
        border: 1px solid rgba(180,83,9,0.20);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 13px;
        color: #92400e !important;
    }
    .result-box {
        background: linear-gradient(135deg, rgba(26,127,55,0.07), rgba(2,132,199,0.04));
        border: 1px solid rgba(26,127,55,0.25);
        border-radius: 12px;
        padding: 24px 28px;
        margin: 16px 0;
        text-align: center;
    }
    .result-value {
        font-size: 52px !important;
        font-weight: 800 !important;
        color: #1a7f37 !important;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -2px;
    }
    .result-unit {
        font-size: 20px;
        color: #57606a !important;
        font-weight: 500;
    }
    .result-label {
        font-size: 13px;
        color: #57606a !important;
        margin-top: 6px;
    }

    /* ── SIDEBAR ────────────────────────────────────────────────── */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background-color: #ffffff !important;
        border-right: 1px solid #d0d7de !important;
    }
    [data-testid="stSidebar"] * { color: #1c2128 !important; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label { color: #57606a !important; }

    /* ── DATAFRAME ──────────────────────────────────────────────── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    [data-testid="stDataFrame"] * {
        background-color: #ffffff !important;
        color: #1c2128 !important;
    }

    /* ── TOMBOL ─────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #1a7f37, #2da44e) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        padding: 10px 28px;
        font-weight: 700;
        font-size: 14px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2da44e, #3fb950) !important;
        transform: translateY(-1px);
    }

    /* ── TAB ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: #eaeef2 !important;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #d0d7de !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        color: #57606a !important;
        font-weight: 600 !important;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #1c2128 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    /* Tab panel body */
    [data-baseweb="tab-panel"] { background-color: transparent !important; }

    /* ── NUMBER INPUT ────────────────────────────────────────────── */
    .stNumberInput input,
    .stNumberInput [data-baseweb="input"],
    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        border: 1px solid #d0d7de !important;
        color: #1c2128 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    /* +/- spinner buttons */
    .stNumberInput button {
        background-color: #f6f8fa !important;
        border-color: #d0d7de !important;
        color: #1c2128 !important;
    }

    /* ── ALERT / SUCCESS / WARNING ──────────────────────────────── */
    [data-testid="stAlert"] {
        background-color: #f0f9ff !important;
        color: #0550ae !important;
        border-color: #bae6fd !important;
    }

    /* ── SPINNER ────────────────────────────────────────────────── */
    [data-testid="stSpinner"] * { color: #57606a !important; }

    /* ── DIVIDER ─────────────────────────────────────────────────── */
    hr { border-color: #d0d7de !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

SAVE_DIR  = r"d:\rainfall-forecasting\output_pengujian"
CKPT_PATH = os.path.join(SAVE_DIR, "checkpoint.pkl")
DATA_PATH = r"d:\rainfall-forecasting\df_imputed.csv"

FEATURES   = ["suhu", "kelembapan", "curah_hujan"]
TIMESTEP   = 7
N_FEATURES = 3

def load_checkpoint():
    """Muat seluruh checkpoint dari file pkl."""
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "rb") as f:
            return pickle.load(f)
    return None

def get_color(metric, value):
    """Warna berdasarkan nilai metrik."""
    if metric == "RMSE":
        return "metric-good" if value < 10 else "metric-ok"
    if metric == "MAE":
        return "metric-good" if value < 8 else "metric-ok"
    if metric in ("NSE", "R2"):
        return "metric-good" if value > 0.7 else "metric-ok"
    return ""

def build_model_from_params(n1, n2, lr):
    """Bangun model LSTM dari parameter gbest."""
    model = Sequential([
        Input(shape=(TIMESTEP, N_FEATURES)),
        LSTM(int(n1), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        LSTM(int(n2)),
        tf.keras.layers.Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=float(lr)), loss="mse")
    return model

def plotly_theme():
    """Tema Plotly light seragam (tanpa xaxis/yaxis/legend agar bisa di-override)."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Plus Jakarta Sans", color="#57606a", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
    )

def html_table(df, highlight_min=None, highlight_max=None, fmt=None):
    """Render DataFrame sebagai tabel HTML light-theme sepenuhnya."""
    fmt = fmt or {}

    # Tentukan sel highlight
    hl_cells = set()
    if highlight_min:
        for col in highlight_min:
            if col in df.columns:
                idx = df[col].idxmin()
                hl_cells.add((idx, col))
    if highlight_max:
        for col in highlight_max:
            if col in df.columns:
                idx = df[col].idxmax()
                hl_cells.add((idx, col))

    # Bangun HTML
    header_html = "<tr><th>#</th>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    rows_html = ""
    for i, (idx, row) in enumerate(df.iterrows()):
        row_bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        cells = f"<td style='color:#57606a;font-size:12px;'>{i+1}</td>"
        for col in df.columns:
            val = row[col]
            fmt_val = fmt.get(col, "{}").format(val) if col in fmt else str(val)
            is_hl = (idx, col) in hl_cells
            bg = "#bbf7d0" if is_hl else "transparent"
            fw = "700" if is_hl else "400"
            color = "#1a7f37" if is_hl else "#1c2128"
            cells += f"<td style='background:{bg};color:{color};font-weight:{fw};'>{fmt_val}</td>"
        rows_html += f"<tr style='background:{row_bg};'>{cells}</tr>"

    return f"""
    <div style='overflow-x:auto;border:1px solid #d0d7de;border-radius:10px;margin:4px 0;'>
    <table style='width:100%;border-collapse:collapse;font-family:Plus Jakarta Sans,sans-serif;font-size:13px;'>
      <thead style='background:#f0f6ff;'>
        <tr><th style='padding:10px 12px;text-align:left;color:#0550ae;font-size:11px;
                       text-transform:uppercase;letter-spacing:.6px;border-bottom:1px solid #d0d7de;'>#</th>
            {"".join(f"<th style='padding:10px 12px;text-align:left;color:#0550ae;font-size:11px;text-transform:uppercase;letter-spacing:.6px;border-bottom:1px solid #d0d7de;'>{c}</th>" for c in df.columns)}
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table></div>
    """

def apply_axis_style(fig):
    """Terapkan gaya axis dan legend default pada figure."""
    fig.update_xaxes(gridcolor="#e5e7eb", linecolor="#d0d7de", tickfont=dict(color="#57606a"))
    fig.update_yaxes(gridcolor="#e5e7eb", linecolor="#d0d7de", tickfont=dict(color="#57606a"))
    fig.update_layout(legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#d0d7de", borderwidth=1))
    return fig


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Konfigurasi")
    st.markdown("---")

    st.markdown("**Status Checkpoint**")

    ckpt_data = load_checkpoint()
    if ckpt_data:
        keys = list(ckpt_data.keys())
        pso_done  = [k for k in keys if k.startswith("PSO") and not k.startswith("m_")]
        model_done = [k for k in keys if k.startswith("m_")]
        st.success(f"{len(pso_done)} skenario PSO selesai")
        st.success(f"{len(model_done)} model terlatih")
        with st.expander("Lihat semua key"):
            for k in keys:
                st.code(k, language=None)
    else:
        st.warning("Checkpoint tidak ditemukan.")

    st.markdown("---")
    st.caption("LSTM-PSO | Curah Hujan Pasuruan")
    st.caption("BMKG Stasiun Geofisika | 2021–2025")


# ═══════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">LSTM-PSO • Rainfall Forecasting</div>
    <h1 class="hero-title">Prediksi Curah Hujan Kabupaten Pasuruan</h1>
    <p class="hero-sub">BMKG Stasiun Geofisika Pasuruan &nbsp;|&nbsp; Data 2021–2025 &nbsp;|&nbsp; Model LSTM dioptimasi dengan Particle Swarm Optimization</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "Dashboard Hasil",
    "Grafik Prediksi",
    "Prediksi Baru",
])


# ───────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD HASIL
# ───────────────────────────────────────────────────────────────────
with tab1:
    if ckpt_data is None:
        st.markdown("""
        <div class="warning-box">
        <strong>Checkpoint belum ditemukan.</strong> Jalankan terlebih dahulu notebook 
        <code>Skenario_pengujian.ipynb</code> hingga selesai, lalu sesuaikan path di sidebar.
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Kumpulkan semua metrik dari checkpoint ──
        all_metrics = []
        model_keys = {
            "LSTM1_70":  "LSTM1 (70:30)",
            "LSTM2_80":  "LSTM2 (80:20)",
            "m_PSO1":    "LSTM-PSO1",
            "m_PSO2":    "LSTM-PSO2",
            "m_PSO3":    "LSTM-PSO3",
            "m_PSO4":    "LSTM-PSO4",
            "m_PSO5":    "LSTM-PSO5",
            "m_PSO6":    "LSTM-PSO6",
            "m_PSO7":    "LSTM-PSO7",
            "m_PSO8":    "LSTM-PSO8",
            "m_PSO9":    "LSTM-PSO9",
            "ARIMA":     "ARIMA",
            "TES":       "TES (Holt-Winters)",
            "GRU":       "GRU",
        }
        for key, label in model_keys.items():
            if key in ckpt_data and isinstance(ckpt_data[key], dict):
                m = ckpt_data[key].get("metrics", ckpt_data[key])
                if "RMSE" in m:
                    all_metrics.append({
                        "Model": label,
                        "RMSE":  round(m.get("RMSE", 0), 4),
                        "MAE":   round(m.get("MAE", 0), 4),
                        "NSE":   round(m.get("NSE", 0), 4),
                        "R²":    round(m.get("R2", 0), 4),
                    })

        if not all_metrics:
            st.info("Belum ada metrik evaluasi di checkpoint. Pastikan skenario sudah selesai dijalankan.")
        else:
            df_metrics = pd.DataFrame(all_metrics).sort_values("RMSE").reset_index(drop=True)
            best = df_metrics.iloc[0]

            # ── 4 Metric Cards model terbaik ──
            st.markdown('<div class="section-title">Model Terbaik</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            for col, metric, val in [
                (c1, "RMSE", best["RMSE"]),
                (c2, "MAE",  best["MAE"]),
                (c3, "NSE",  best["NSE"]),
                (c4, "R²",   best["R²"]),
            ]:
                cls = get_color(metric, val)
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{metric}</div>
                        <div class="metric-value {cls}">{val:.4f}</div>
                        <div class="metric-model">{best['Model']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("&nbsp;")

            # ── Tabel perbandingan semua model ──
            col_tbl, col_chart = st.columns([1, 1], gap="large")

            with col_tbl:
                st.markdown('<div class="section-title">Perbandingan Semua Model</div>', unsafe_allow_html=True)
                df_show = df_metrics.copy().reset_index(drop=True)
                st.markdown(
                    html_table(
                        df_show,
                        highlight_min=["RMSE","MAE"],
                        highlight_max=["NSE","R²"],
                        fmt={"RMSE":"{:.4f}","MAE":"{:.4f}","NSE":"{:.4f}","R²":"{:.4f}"},
                    ),
                    unsafe_allow_html=True,
                )

            with col_chart:
                st.markdown('<div class="section-title">Bar Chart Metrik</div>', unsafe_allow_html=True)
                metric_sel = st.selectbox("Pilih metrik", ["RMSE","MAE","NSE","R²"], key="metric_bar")
                ascending = metric_sel in ["RMSE","MAE"]
                df_chart = df_metrics.sort_values(metric_sel, ascending=ascending)

                colors = ["#1a7f37" if v == df_chart[metric_sel].min() else "#0969da"
                          for v in df_chart[metric_sel]] if ascending else \
                         ["#1a7f37" if v == df_chart[metric_sel].max() else "#0969da"
                          for v in df_chart[metric_sel]]

                fig_bar = go.Figure(go.Bar(
                    x=df_chart["Model"], y=df_chart[metric_sel],
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in df_chart[metric_sel]],
                    textposition="outside", textfont=dict(size=10, color="#57606a"),
                ))

                fig_bar.update_layout(
                    **plotly_theme(),
                    height=350,
                    showlegend=False,
                )
                apply_axis_style(fig_bar)
                fig_bar.update_layout(yaxis_title=metric_sel)
                fig_bar.update_xaxes(tickangle=-35, tickfont=dict(size=10))
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── PSO Parameter Log ──
            pso_keys_log = [k for k in ckpt_data if k.startswith("PSO") and not k.startswith("m_")]
            if pso_keys_log:
                st.markdown('<div class="section-title">Log Parameter PSO</div>', unsafe_allow_html=True)
                pso_rows = []
                for k in sorted(pso_keys_log):
                    d = ckpt_data[k]
                    log = d.get("log", {})
                    if log:
                        pso_rows.append({
                            "Skenario": log.get("Skenario",""),
                            "w":        log.get("w",""),
                            "c1":       log.get("c1",""),
                            "c2":       log.get("c2",""),
                            "Swarm":    log.get("swarm",""),
                            "Iterasi":  log.get("iter",""),
                            "Fitness":  round(float(d.get("gfit", 0)), 5),
                        })
                if pso_rows:
                    df_pso = pd.DataFrame(pso_rows)
                    st.markdown(
                        html_table(
                            df_pso,
                            fmt={"Fitness":"{:.5f}"},
                        ),
                        unsafe_allow_html=True,
                    )


# ───────────────────────────────────────────────────────────────────
# TAB 2 — GRAFIK PREDIKSI
# ───────────────────────────────────────────────────────────────────
with tab2:
    if ckpt_data is None:
        st.warning("Checkpoint belum ditemukan. Sesuaikan path di sidebar.")
    else:
        # Ambil data aktual dari checkpoint (LSTM2_80 sebagai referensi)
        pred_map = {}
        actual   = None

        src_keys = {
            "LSTM1_70": "LSTM1 (70:30)",
            "LSTM2_80": "LSTM2 (80:20)",
            "m_PSO1": "LSTM-PSO1", "m_PSO2": "LSTM-PSO2",
            "m_PSO3": "LSTM-PSO3", "m_PSO4": "LSTM-PSO4",
            "m_PSO5": "LSTM-PSO5", "m_PSO6": "LSTM-PSO6",
            "m_PSO7": "LSTM-PSO7", "m_PSO8": "LSTM-PSO8",
            "m_PSO9": "LSTM-PSO9",
            "ARIMA": "ARIMA", "TES": "TES (Holt-Winters)", "GRU": "GRU",
        }

        for key, label in src_keys.items():
            if key in ckpt_data:
                d = ckpt_data[key]
                pred = d.get("pred")
                if pred is not None:
                    pred_map[label] = np.array(pred)

        # Load data aktual
        if os.path.exists(DATA_PATH):
            df_raw = pd.read_csv(DATA_PATH)
            df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"], errors="coerce")
            df_raw.set_index("tanggal", inplace=True)
            n_total = len(df_raw) - TIMESTEP
            n_test  = int(n_total * 0.2)
            actual  = df_raw["curah_hujan"].values[-(n_test):]
            dates   = df_raw.index[-(n_test):]
        else:
            # Pakai prediksi pertama sebagai panjang referensi
            if pred_map:
                first_pred = next(iter(pred_map.values()))
                actual = np.zeros(len(first_pred))
                dates  = pd.date_range(end="2025-12-31", periods=len(first_pred), freq="D")

        if not pred_map:
            st.info("Belum ada data prediksi di checkpoint.")
        else:
            st.markdown('<div class="section-title">Grafik Aktual vs Prediksi</div>', unsafe_allow_html=True)

            # Pilih model yang ditampilkan
            all_model_names = list(pred_map.keys())
            selected_models = st.multiselect(
                "Pilih model yang ditampilkan",
                options=all_model_names,
                default=all_model_names[:4] if len(all_model_names) >= 4 else all_model_names,
            )

            if selected_models:
                COLOR_PALETTE = [
                    "#0284c7","#1a7f37","#f97316","#7c3aed",
                    "#ea580c","#059669","#2563eb","#c026d3",
                    "#ca8a04","#dc2626","#64748b",
                ]

                fig = go.Figure()

                # Aktual
                if actual is not None and actual.any():
                    fig.add_trace(go.Scatter(
                        x=dates, y=actual,
                        name="Aktual",
                        line=dict(color="#374151", width=1.8, dash="dot"),
                        opacity=0.9,
                    ))

                for i, name in enumerate(selected_models):
                    pred = pred_map[name]
                    n = min(len(pred), len(dates))
                    fig.add_trace(go.Scatter(
                        x=dates[:n], y=pred[:n],
                        name=name,
                        line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=1.4),
                        opacity=0.85,
                    ))

                fig.update_layout(
                    **plotly_theme(),
                    height=420,
                    hovermode="x unified",
                )
                apply_axis_style(fig)
                fig.update_layout(
                    xaxis_title="Tanggal",
                    yaxis_title="Curah Hujan (mm)",
                    legend=dict(
                        bgcolor="rgba(255,255,255,0.95)", bordercolor="#d0d7de", borderwidth=1,
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Grafik residual (error) ──
            if len(selected_models) == 1 and actual is not None and actual.any():
                name = selected_models[0]
                pred = pred_map[name]
                n = min(len(pred), len(actual))
                residual = actual[:n] - pred[:n]

                st.markdown('<div class="section-title">Residual (Aktual − Prediksi)</div>', unsafe_allow_html=True)
                fig_res = go.Figure()
                fig_res.add_trace(go.Bar(
                    x=dates[:n], y=residual,
                    marker_color=["#1a7f37" if v >= 0 else "#dc2626" for v in residual],
                    name="Residual", opacity=0.75,
                ))
                fig_res.add_hline(y=0, line_color="#9ca3af", line_dash="dot")
                fig_res.update_layout(**plotly_theme(), height=260)
                apply_axis_style(fig_res)
                fig_res.update_layout(xaxis_title="Tanggal", yaxis_title="Residual (mm)")
                st.plotly_chart(fig_res, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 3 — PREDIKSI BARU
# ───────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="info-box">
    <strong>Cara penggunaan:</strong> Masukkan data cuaca <strong>7 hari terakhir</strong> 
    (suhu, kelembapan, curah hujan), lalu klik <em>Prediksi</em>. Model akan mengestimasi 
    curah hujan hari berikutnya.
    </div>
    """, unsafe_allow_html=True)

    # ── Pilih model yang digunakan ──
    st.markdown('<div class="section-title">Pilih Model</div>', unsafe_allow_html=True)

    model_choice = None
    gbest_params = None

    if ckpt_data:
        pso_options = {k: k for k in ckpt_data if k.startswith("PSO") and not k.startswith("m_")}
        baseline_options = {"LSTM1_70": "LSTM1 (70:30)", "LSTM2_80": "LSTM2 (80:20)"}

        available = {}
        for k, label in {**baseline_options, **{k: k for k in pso_options}}.items():
            if k in ckpt_data and "gbest" in ckpt_data.get(k, {}):
                available[k] = label

        if available:
            model_choice = st.selectbox(
                "Gunakan parameter dari skenario:",
                options=list(available.keys()),
                format_func=lambda k: available[k],
            )
            if model_choice and model_choice in ckpt_data:
                gbest_params = ckpt_data[model_choice].get("gbest")
                if gbest_params is not None:
                    n1, n2, lr, batch = (
                        int(gbest_params[0]), int(gbest_params[1]),
                        float(gbest_params[2]), int(gbest_params[3])
                    )
                    st.markdown(f"""
                    <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;
                                padding:12px 18px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#0284c7;">
                    N1={n1} &nbsp;|&nbsp; N2={n2} &nbsp;|&nbsp; LR={lr:.5f} &nbsp;|&nbsp; Batch={batch}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Belum ada parameter model di checkpoint. Jalankan notebook terlebih dahulu.")
    else:
        st.warning("Checkpoint tidak ditemukan. Sesuaikan path di sidebar.")

    # ── Muat scaler dari checkpoint ──
    scaler = ckpt_data.get("scaler") if ckpt_data else None
    if scaler is None:
        st.markdown("""
        <div class="warning-box">
        <strong>Scaler belum ditemukan di checkpoint.</strong>
        Tambahkan <code>ckpt_save('scaler', scaler)</code> di Section 5 notebook lalu run ulang cell tersebut.
        Sementara itu prediksi menggunakan batas normalisasi manual.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Input Data 7 Hari Terakhir</div>', unsafe_allow_html=True)

    # Header kolom
    h0, h1, h2, h3 = st.columns([0.5, 1, 1, 1])
    h0.markdown("**Hari**")
    h1.markdown("**Suhu (°C)**")
    h2.markdown("**Kelembapan (%)**")
    h3.markdown("**Curah Hujan (mm)**")

    input_data = []
    for i in range(TIMESTEP):
        c0, c1, c2, c3 = st.columns([0.5, 1, 1, 1])
        c0.markdown(f"<div style='padding-top:8px;color:#57606a;font-size:13px;'>H-{TIMESTEP-i}</div>",
                    unsafe_allow_html=True)
        suhu  = c1.number_input("", min_value=15.0, max_value=45.0, value=28.0, step=0.1,
                                 key=f"suhu_{i}", label_visibility="collapsed")
        kelm  = c2.number_input("", min_value=30.0, max_value=100.0, value=75.0, step=0.5,
                                 key=f"kelm_{i}", label_visibility="collapsed")
        hujan = c3.number_input("", min_value=0.0, max_value=500.0, value=0.0, step=0.5,
                                 key=f"hujan_{i}", label_visibility="collapsed")
        input_data.append([suhu, kelm, hujan])

    st.markdown("&nbsp;")

    # ── Tombol prediksi ──
    col_btn, col_empty = st.columns([1, 2])
    with col_btn:
        predict_btn = st.button("Prediksi Curah Hujan", type="primary")

    if predict_btn:
        if not TF_AVAILABLE:
            st.error("TensorFlow tidak terinstall. Jalankan: `pip install tensorflow`")
        elif gbest_params is None:
            st.error("Parameter model belum tersedia. Pastikan checkpoint sudah ada.")
        else:
            with st.spinner("Membangun & menjalankan model..."):
                try:
                    arr = np.array(input_data, dtype=float)   # shape (7, 3)

                    # ── Normalisasi ──────────────────────────────────────
                    if scaler is not None:
                        # Gunakan scaler asli dari notebook (akurat)
                        arr_norm = scaler.transform(arr)
                    else:
                        # Fallback: normalisasi manual jika scaler belum disimpan
                        mins = np.array([15.0,  30.0,   0.0])
                        maxs = np.array([40.0, 100.0, 200.0])
                        arr_norm = (arr - mins) / (maxs - mins)
                        arr_norm = np.clip(arr_norm, 0, 1)

                    X_input = arr_norm.reshape(1, TIMESTEP, N_FEATURES)

                    tf.keras.backend.clear_session()
                    model = build_model_from_params(n1, n2, lr)

                    pred_norm = float(model.predict(X_input, verbose=0)[0][0])

                    # ── Denormalisasi ────────────────────────────────────
                    if scaler is not None:
                        # Rekonstruksi array dummy lalu inverse_transform
                        dummy = np.zeros((1, N_FEATURES))
                        dummy[0, -1] = pred_norm          # kolom terakhir = curah_hujan
                        pred_mm = float(scaler.inverse_transform(dummy)[0, -1])
                    else:
                        pred_mm = pred_norm * (200.0 - 0.0) + 0.0

                    pred_mm = max(0.0, pred_mm)

                    # ── Tampilkan hasil ──
                    if pred_mm < 5:
                        kategori, emoji, warna = "Tidak Hujan / Sangat Ringan", "☀️", "#b45309"
                    elif pred_mm < 20:
                        kategori, emoji, warna = "Hujan Ringan", "🌦️", "#0284c7"
                    elif pred_mm < 50:
                        kategori, emoji, warna = "Hujan Sedang", "🌧️", "#1d4ed8"
                    elif pred_mm < 100:
                        kategori, emoji, warna = "Hujan Lebat", "⛈️", "#6d28d9"
                    else:
                        kategori, emoji, warna = "Hujan Sangat Lebat", "🌊", "#dc2626"

                    st.markdown(f"""
                    <div class="result-box">
                        <div style="font-size:40px;margin-bottom:8px">{emoji}</div>
                        <div>
                            <span class="result-value">{pred_mm:.1f}</span>
                            <span class="result-unit"> mm</span>
                        </div>
                        <div style="color:{warna};font-weight:700;font-size:15px;margin-top:8px;">
                            {kategori}
                        </div>
                        <div class="result-label">Estimasi curah hujan hari berikutnya</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Info sumber normalisasi
                    src = "MinMaxScaler dari checkpoint (akurat)" if scaler is not None \
                          else "Normalisasi manual / fallback"
                    icon = "✅" if scaler is not None else "⚠️"
                    st.markdown(f"""
                    <div style="background:#f8fafc;border:1px solid #d0d7de;border-radius:8px;
                                padding:10px 16px;font-size:12px;color:#57606a;margin-top:8px;">
                        {icon} <strong>Normalisasi:</strong> {src} &nbsp;|&nbsp;
                        <strong>Model:</strong> {model_choice} &nbsp;|&nbsp;
                        <strong>N1={n1}, N2={n2}, LR={lr:.5f}</strong>
                    </div>
                    """, unsafe_allow_html=True)


                    # ── Mini chart input ──
                    st.markdown('<div class="section-title">Data Input 7 Hari Terakhir</div>',
                                unsafe_allow_html=True)
                    df_input = pd.DataFrame(input_data,
                                            columns=["Suhu (°C)", "Kelembapan (%)", "Curah Hujan (mm)"],
                                            index=[f"H-{TIMESTEP-i}" for i in range(TIMESTEP)])

                    fig_in = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                           subplot_titles=["Suhu (°C)", "Kelembapan (%)", "Curah Hujan (mm)"],
                                           vertical_spacing=0.08)
                    colors_in = ["#f97316", "#0284c7", "#1d4ed8"]
                    for row, col in enumerate(df_input.columns):
                        fig_in.add_trace(go.Scatter(
                            x=df_input.index, y=df_input[col],
                            name=col, line=dict(color=colors_in[row], width=2),
                            fill="tozeroy", fillcolor=colors_in[row].replace(")", ",0.08)").replace("rgb","rgba"),
                        ), row=row+1, col=1)
                    fig_in.update_layout(**plotly_theme(), height=340, showlegend=False)
                    apply_axis_style(fig_in)
                    for i in range(1, 4):
                        fig_in.update_xaxes(gridcolor="#e5e7eb", row=i, col=1)
                        fig_in.update_yaxes(gridcolor="#e5e7eb", row=i, col=1)
                    st.plotly_chart(fig_in, use_container_width=True)

                except Exception as e:
                    st.error(f"Error saat prediksi: {e}")