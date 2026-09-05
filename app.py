"""
Sonar Intelligence Dashboard — Underwater Marine Debris & Target Detection
"""
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st
import cv2, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from PIL import Image
import io, base64, time, json
from datetime import datetime, timezone

st.set_page_config(
    page_title="Marine Sonar Intelligence",
    page_icon="🌊", layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Deep Ocean Sonar Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
  --bg-primary:   #06141F;
  --bg-secondary: #0A2230;
  --bg-card:      #0D2B38;
  --border:       #1B5263;
  --accent:       #19C3D1;
  --accent2:      #4DD0E1;
  --text:         #E8F7FA;
  --text-sec:     #91B7C0;
  --success:      #38D39F;
  --warning:      #F4C95D;
  --danger:       #FF667A;
  --anomaly:      #B78CFF;
}

/* ── Global reset ── */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg-primary) !important;
  color: var(--text) !important;
}
.main { background-color: var(--bg-primary) !important; }
section[data-testid="stSidebar"] {
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Underwater particle background ── */
.stApp::before {
  content: '';
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background:
    radial-gradient(ellipse at 20% 80%, #0A2A3A44 0%, transparent 60%),
    radial-gradient(ellipse at 80% 20%, #062035 0%, transparent 50%),
    linear-gradient(180deg, #06141F 0%, #08192A 50%, #06141F 100%);
  pointer-events: none; z-index: 0;
  animation: oceanShift 20s ease-in-out infinite alternate;
}
@keyframes oceanShift {
  0%   { opacity: 0.8; }
  100% { opacity: 1.0; }
}

/* ── Header ── */
.sonar-header {
  background: linear-gradient(90deg, #06141F 0%, #0D2B38 40%, #0A2230 70%, #06141F 100%);
  border-bottom: 2px solid var(--accent);
  padding: 0.9rem 1.5rem;
  margin: -1rem -1rem 0.5rem -1rem;
  display: flex; align-items: center; justify-content: space-between;
}
.header-left { display: flex; flex-direction: column; }
.header-title {
  font-size: 1.25rem; font-weight: 700; color: var(--accent);
  letter-spacing: 0.04em; margin: 0;
}
.header-sub {
  font-size: 0.72rem; color: var(--text-sec);
  letter-spacing: 0.08em; text-transform: uppercase; margin: 0;
}
.header-right { display: flex; align-items: center; gap: 1.2rem; }
.status-online {
  display: flex; align-items: center; gap: 0.4rem;
  font-size: 0.75rem; color: var(--success); font-weight: 600;
}
.pulse-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 0 0 rgba(56,211,159,0.6);
  animation: sonarPulse 2s ease-out infinite;
  display: inline-block;
}
@keyframes sonarPulse {
  0%   { box-shadow: 0 0 0 0 rgba(56,211,159,0.6); }
  70%  { box-shadow: 0 0 0 8px rgba(56,211,159,0); }
  100% { box-shadow: 0 0 0 0 rgba(56,211,159,0); }
}
.badge-sih {
  background: rgba(25,195,209,0.12); border: 1px solid rgba(25,195,209,0.4);
  color: var(--accent); padding: 2px 10px; border-radius: 20px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
}
.badge-demo {
  background: rgba(244,201,93,0.12); border: 1px solid rgba(244,201,93,0.4);
  color: var(--warning); padding: 2px 10px; border-radius: 20px;
  font-size: 0.68rem; font-weight: 600;
}

/* ── Sidebar nav ── */
.nav-label {
  font-size: 0.62rem; color: var(--text-sec);
  text-transform: uppercase; letter-spacing: 0.1em;
  padding: 0.5rem 0 0.25rem 0.2rem; margin: 0;
}

/* ── Metric overrides ── */
[data-testid="stMetricValue"] {
  color: var(--accent) !important;
  font-size: 1.6rem !important;
  font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
  color: var(--text-sec) !important;
  font-size: 0.72rem !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 1rem !important;
}

/* ── Cards ── */
.sonar-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.1rem;
  margin-bottom: 0.6rem;
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.sonar-card:hover {
  border-color: var(--accent);
  box-shadow: 0 0 18px rgba(25,195,209,0.08);
}
.sonar-card-anomaly {
  background: linear-gradient(135deg, #120820, #0D1B2A);
  border-color: rgba(183,140,255,0.4);
  box-shadow: 0 0 12px rgba(183,140,255,0.06);
  animation: anomalyBorder 3s ease-in-out infinite alternate;
}
@keyframes anomalyBorder {
  0%   { border-color: rgba(183,140,255,0.3); }
  100% { border-color: rgba(183,140,255,0.7); }
}
.sonar-card-high {
  border-left: 3px solid var(--danger);
}
.sonar-card-medium {
  border-left: 3px solid var(--warning);
}
.sonar-card-low {
  border-left: 3px solid var(--success);
}

/* ── Badges ── */
.sev { padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; display: inline-block; }
.sev-HIGH    { background: rgba(255,102,122,0.15); border: 1px solid var(--danger);  color: var(--danger); }
.sev-MEDIUM  { background: rgba(244,201,93,0.15);  border: 1px solid var(--warning); color: var(--warning); }
.sev-LOW     { background: rgba(56,211,159,0.15);  border: 1px solid var(--success); color: var(--success); }
.sev-UNKNOWN { background: rgba(145,183,192,0.12); border: 1px solid var(--text-sec); color: var(--text-sec); }
.mode-badge {
  padding: 1px 8px; border-radius: 4px; font-size: 0.68rem;
  font-family: 'JetBrains Mono', monospace; display: inline-block;
}
.mode-DEMO { background: rgba(244,201,93,0.15); border: 1px solid rgba(244,201,93,0.4); color: var(--warning); }
.mode-REAL { background: rgba(56,211,159,0.15); border: 1px solid rgba(56,211,159,0.4); color: var(--success); }

/* ── Score bars ── */
.score-row { margin: 4px 0; }
.score-label { font-size: 0.72rem; color: var(--text-sec); margin-bottom: 2px; }
.score-track {
  background: rgba(27,82,99,0.4); border-radius: 3px; height: 6px;
  overflow: hidden; position: relative;
}
.score-fill {
  height: 100%; border-radius: 3px;
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

/* ── Pipeline stages ── */
.pipeline-wrap { display: flex; align-items: center; gap: 0; margin: 0.5rem 0; }
.pipe-step {
  flex: 1; text-align: center; padding: 0.5rem 0.2rem;
  background: var(--bg-card); border: 1px solid var(--border);
  font-size: 0.65rem; color: var(--text-sec);
  transition: all 0.3s ease; position: relative;
}
.pipe-step:first-child { border-radius: 6px 0 0 6px; }
.pipe-step:last-child  { border-radius: 0 6px 6px 0; }
.pipe-step.active  { background: rgba(25,195,209,0.15); border-color: var(--accent); color: var(--accent); font-weight: 600; }
.pipe-step.done    { background: rgba(56,211,159,0.08); border-color: rgba(56,211,159,0.3); color: var(--success); }
.pipe-arrow { color: var(--border); font-size: 0.8rem; padding: 0 1px; }

/* ── Sonar animation ── */
.sonar-ring-wrap {
  display: flex; justify-content: center; align-items: center;
  height: 120px; margin: 0.5rem 0;
  position: relative;
}
.sonar-ring {
  width: 90px; height: 90px; border-radius: 50%;
  border: 2px solid var(--accent);
  position: absolute;
  animation: sonarExpand 3s ease-out infinite;
  opacity: 0;
}
.sonar-ring:nth-child(2) { animation-delay: 1s; }
.sonar-ring:nth-child(3) { animation-delay: 2s; }
.sonar-center {
  width: 12px; height: 12px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 12px var(--accent);
  z-index: 1;
}
@keyframes sonarExpand {
  0%   { width: 12px; height: 12px; opacity: 0.8; border-color: var(--accent); }
  100% { width: 100px; height: 100px; opacity: 0; border-color: rgba(25,195,209,0.1); }
}

/* ── Evidence fusion ── */
.fusion-signal {
  text-align: center; padding: 0.5rem;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 8px; font-size: 0.72rem; color: var(--text-sec);
  transition: all 0.3s ease;
}
.fusion-signal.loaded {
  border-color: var(--accent); color: var(--accent);
  box-shadow: 0 0 10px rgba(25,195,209,0.1);
}
.fusion-result {
  text-align: center; padding: 0.8rem;
  background: linear-gradient(135deg, #0D2B38, #0A2230);
  border: 2px solid var(--accent); border-radius: 10px;
  box-shadow: 0 0 20px rgba(25,195,209,0.12);
}

/* ── Detection result panel ── */
.result-panel {
  background: linear-gradient(135deg, #0D2B38, #0A2230);
  border: 1px solid var(--accent);
  border-radius: 12px; padding: 1.4rem;
  box-shadow: 0 0 30px rgba(25,195,209,0.08);
}
.result-title {
  font-size: 1rem; font-weight: 700; color: var(--accent);
  text-transform: uppercase; letter-spacing: 0.06em;
  border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; margin-bottom: 0.8rem;
}
.result-row { display: flex; justify-content: space-between; margin: 0.3rem 0; }
.result-key { font-size: 0.78rem; color: var(--text-sec); }
.result-val { font-size: 0.78rem; color: var(--text); font-weight: 600; font-family: 'JetBrains Mono', monospace; }

/* ── Image captions ── */
.img-label {
  background: rgba(25,195,209,0.1); border: 1px solid rgba(25,195,209,0.2);
  color: var(--accent); text-align: center; padding: 3px 0;
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em;
  border-radius: 0 0 6px 6px; margin-top: -4px;
}

/* ── Buttons ── */
.stButton > button {
  background: rgba(25,195,209,0.1) !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  background: rgba(25,195,209,0.22) !important;
  box-shadow: 0 0 14px rgba(25,195,209,0.2) !important;
}
button[kind="primary"] {
  background: linear-gradient(90deg, #19C3D1, #0FA8B4) !important;
  border: none !important; color: #06141F !important;
  font-weight: 700 !important;
}
button[kind="primary"]:hover {
  box-shadow: 0 0 20px rgba(25,195,209,0.35) !important;
}

/* ── Inputs ── */
.stSelectbox > div, .stTextInput > div {
  background: var(--bg-secondary) !important;
  border-color: var(--border) !important;
}
.stRadio > div { color: var(--text) !important; }

/* ── Expanders ── */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Section headers ── */
.sec-hdr {
  font-size: 0.92rem; font-weight: 700; color: var(--accent2);
  text-transform: uppercase; letter-spacing: 0.06em;
  border-left: 3px solid var(--accent); padding-left: 0.6rem;
  margin: 0.6rem 0 0.4rem 0;
}

/* ── Toast / info ── */
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def img_bytes(arr: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr.ndim == 3 else arr
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()

def show_sonar(arr: np.ndarray, label: str = "", key: str = ""):
    if arr is None:
        st.warning("Image unavailable"); return
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr.ndim == 3 else arr
    st.image(rgb, use_container_width=True)
    if label:
        st.markdown(f'<div class="img-label">{label}</div>', unsafe_allow_html=True)

def sev_badge(s: str) -> str:
    return f'<span class="sev sev-{s}">{s}</span>'

def mode_badge(m: str) -> str:
    return f'<span class="mode-badge mode-{m}">MODE: {m}</span>'

def score_bar(val: float, color: str, label: str) -> str:
    pct = min(100, max(0, val * 100))
    return f"""<div class="score-row">
        <div class="score-label">{label}: <b style="color:{color}">{pct:.0f}%</b></div>
        <div class="score-track">
          <div class="score-fill" style="width:{pct:.0f}%;background:{color};"></div>
        </div></div>"""

def sonar_ring() -> str:
    return """<div class="sonar-ring-wrap">
        <div class="sonar-ring"></div>
        <div class="sonar-ring"></div>
        <div class="sonar-ring"></div>
        <div class="sonar-center"></div>
    </div>"""

def pipeline_bar(stage_idx: int) -> str:
    stages = ["INPUT","QUALITY","CLAHE","DETECT","SHADOW","ANOMALY","FUSE","GEOTAG","REPORT"]
    html = '<div class="pipeline-wrap">'
    for i, s in enumerate(stages):
        cls = "pipe-step"
        if i < stage_idx:   cls += " done"
        elif i == stage_idx: cls += " active"
        prefix = "✓ " if i < stage_idx else ("▶ " if i == stage_idx else "")
        html += f'<div class="{cls}">{prefix}{s}</div>'
        if i < len(stages)-1: html += '<div class="pipe-arrow">›</div>'
    html += '</div>'
    return html


# ══════════════════════════════════════════════════════════════════════════════
# SERVICES (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initializing AI pipeline...")
def get_pipeline():
    from services.pipeline_service import SonarAnalysisPipeline
    return SonarAnalysisPipeline()

@st.cache_resource(show_spinner=False)
def ensure_db():
    from services.mission_service import init_database
    init_database()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div class="sonar-header">
      <div class="header-left">
        <div class="header-title">🌊 AI-POWERED UNDERWATER SONAR</div>
        <div class="header-sub">Marine Debris &amp; Anomaly Detection System — SIH26057</div>
      </div>
      <div class="header-right">
        <span class="badge-sih">SIH26057</span>
        <span class="badge-demo">⚠ DEMO / SYNTHETIC</span>
        <span class="status-online"><span class="pulse-dot"></span> SYSTEM ONLINE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:0.6rem 0 0.2rem">
          <div style="font-size:1.4rem">🛸</div>
          <div style="font-size:0.7rem;color:#19C3D1;font-weight:700;letter-spacing:0.1em">SONAR SYSTEM</div>
        </div>""", unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state["page"] = "dashboard"

        pages = [
            ("dashboard",   "🏠", "Dashboard"),
            ("analysis",    "📡", "Sonar Analysis"),
            ("detections",  "🎯", "Detections"),
            ("anomalies",   "⚠️", "Anomaly Analysis"),
            ("shadow",      "🌑", "Acoustic Shadow"),
            ("map",         "🗺️", "Geolocation"),
            ("analytics",   "📊", "Analytics"),
            ("missions",    "🚢", "Missions"),
            ("models",      "🤖", "System Status"),
            ("reports",     "📋", "Reports"),
        ]

        st.markdown('<p class="nav-label">Navigation</p>', unsafe_allow_html=True)
        for key, icon, label in pages:
            active = st.session_state.get("page") == key
            btn_type = "primary" if active else "secondary"
            if st.button(f"{icon}  {label}", use_container_width=True,
                         type=btn_type, key=f"nav_{key}"):
                st.session_state["page"] = key
                st.rerun()

        st.divider()

        # Pipeline mode
        try:
            p = get_pipeline()
            mode = p.detector.mode
            st.markdown(f"**Detector:** {mode_badge(mode)}", unsafe_allow_html=True)
        except Exception:
            st.caption("Pipeline not loaded")

        st.divider()
        st.markdown('<p class="nav-label">Data Labels</p>', unsafe_allow_html=True)
        st.caption("All demo data: **DEMO / SYNTHETIC**")
        st.caption("Coordinates: **SIMULATED**")
        st.caption("Evidence scores: **engineering heuristic**")

        st.divider()
        if st.button("🔄 Run Demo Setup", use_container_width=True):
            with st.spinner("Generating demo data..."):
                try:
                    from scripts.setup_demo import setup_demo
                    setup_demo()
                    st.success("Demo setup complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown("## 🏠 Mission Dashboard")

    ensure_db()
    from services.mission_service import get_statistics, get_all_detections
    stats = get_statistics()
    dets  = get_all_detections(limit=100)

    # ── Sonar animation + KPIs ─────────────────────────────────────────────────
    hc1, hc2 = st.columns([1, 5])
    with hc1:
        st.markdown(sonar_ring(), unsafe_allow_html=True)
    with hc2:
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("🚢 Missions",    stats["total_missions"])
        c2.metric("📡 Images",      stats["total_images"])
        c3.metric("🎯 Known",       stats["known_debris"])
        c4.metric("⚠️ Anomalies",   stats["total_anomalies"])
        c5.metric("🔴 High Risk",   stats["high_risk"])
        c6.metric("⚡ Avg Evidence",f"{stats['avg_evidence_pct']:.0f}%")

    st.markdown("---")

    # ── Pipeline display ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">SIH26057 Detection Pipeline</div>', unsafe_allow_html=True)
    st.markdown(pipeline_bar(9), unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    chart_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#91B7C0", size=11),
        margin=dict(t=30, b=30, l=20, r=20),
    )

    with col1:
        st.markdown('<div class="sec-hdr">Class Distribution</div>', unsafe_allow_html=True)
        cd = stats.get("class_distribution", {})
        if cd:
            fig = px.pie(names=list(cd.keys()), values=list(cd.values()),
                         color_discrete_sequence=["#19C3D1","#4DD0E1","#38D39F",
                                                   "#F4C95D","#FF667A","#B78CFF","#91B7C0"],
                         hole=0.45)
            fig.update_traces(textfont_color="white")
            fig.update_layout(**chart_layout, showlegend=True,
                              legend=dict(font=dict(color="#91B7C0", size=9)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No detections yet — run demo setup.")

    with col2:
        st.markdown('<div class="sec-hdr">Severity Distribution</div>', unsafe_allow_html=True)
        sd = stats.get("severity_distribution", {})
        if sd:
            sev_c = {"HIGH":"#FF667A","MEDIUM":"#F4C95D","LOW":"#38D39F","UNKNOWN":"#91B7C0"}
            fig = px.bar(x=list(sd.keys()), y=list(sd.values()),
                         color=list(sd.keys()), color_discrete_map=sev_c)
            fig.update_layout(**chart_layout, showlegend=False,
                              xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No severity data.")

    with col3:
        st.markdown('<div class="sec-hdr">Evidence Score Distribution</div>', unsafe_allow_html=True)
        if dets:
            scores = [d["evidence_score"] for d in dets if d.get("evidence_score") is not None]
            fig = px.histogram(x=scores, nbins=10, color_discrete_sequence=["#19C3D1"])
            fig.update_layout(**chart_layout, xaxis=dict(range=[0,1]),
                              xaxis_title="Evidence Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    # ── Recent detections ──────────────────────────────────────────────────────
    if dets:
        st.markdown("---")
        st.markdown('<div class="sec-hdr">Recent Detections</div>', unsafe_allow_html=True)
        df = pd.DataFrame(dets[:20])
        show_cols = ["detection_id","class_name","confidence","evidence_score","severity","is_anomaly","mode"]
        df_show = df[[c for c in show_cols if c in df.columns]].copy()
        for col in ["confidence","evidence_score"]:
            if col in df_show:
                df_show[col] = df_show[col].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
        st.dataframe(df_show, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Sonar Analysis (MAIN DEMO — complete pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def page_sonar_analysis():
    st.markdown("## 📡 Sonar Analysis")
    st.caption("Complete SIH26057 end-to-end pipeline demonstration")
    ensure_db()

    # ── Source selection ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">1. Select Sonar Image Source</div>', unsafe_allow_html=True)
    src = st.radio("Source:", ["📁 Demo Mission Images", "📤 Upload Image", "🔬 Generate Synthetic"],
                   horizontal=True, label_visibility="collapsed")

    raw_img = None; scenario_type = None; annotations = []
    lat = lon = depth_m = None; img_label = "Unknown"

    if src == "📁 Demo Mission Images":
        demo_dir = Path("data/demo")
        meta_file = demo_dir / "MISSION-001_metadata.json"
        if not meta_file.exists():
            st.warning("Demo images not found. Click '🔄 Run Demo Setup' in the sidebar.")
            return
        with open(meta_file) as f:
            mission_meta = json.load(f)
        images = mission_meta.get("images", [])
        options = {
            f"{i['image_id']} │ {i['scenario_type'].replace('_',' ').title()} │ {i.get('description','')[:35]}": i
            for i in images
        }
        sel_key = st.selectbox("Select demo image:", list(options.keys()))
        sel = options[sel_key]
        raw_img = cv2.imread(sel["filepath"])
        scenario_type = sel.get("scenario_type")
        annotations = sel.get("annotations", [])
        lat, lon, depth_m = sel.get("lat"), sel.get("lon"), sel.get("depth_m")
        img_label = sel.get("description", sel["filename"])
        st.caption(f"⚠️ {sel.get('data_label','DEMO/SYNTHETIC')} — Coordinates: DEMO/SIMULATED")

    elif src == "📤 Upload Image":
        up = st.file_uploader("Upload sonar image (PNG/JPG/TIFF)",
                               type=["png","jpg","jpeg","tiff","bmp"])
        if up:
            arr = np.frombuffer(up.read(), np.uint8)
            raw_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img_label = up.name
        else:
            st.info("📤 Upload a sonar image to analyze.")
            return

    elif src == "🔬 Generate Synthetic":
        from simulation.synthetic_sonar import generate_scenario, DEMO_SCENARIOS
        sc_opts = {s: s for _, s, _ in DEMO_SCENARIOS}
        sel_sc = st.selectbox("Scenario:", list(sc_opts.keys()))
        scenario_type = sc_opts[sel_sc]
        seed = st.number_input("Seed", 0, 9999, 42)
        arr, meta = generate_scenario(int(seed), scenario_type, seed=int(seed))
        raw_img = arr if arr.ndim == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        annotations = meta.get("annotations", [])
        img_label = f"Synthetic — {scenario_type}"
        st.caption("⚠️ SYNTHETIC / DEMO — Not a real sonar measurement")

    if raw_img is None:
        return

    # ── Raw image preview ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-hdr">2. Raw Sonar Image</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns([4, 1])
    with pc1:
        show_sonar(raw_img, "RAW SONAR — No processing applied")
    with pc2:
        h, w = raw_img.shape[:2]
        st.metric("Width", f"{w} px")
        st.metric("Height", f"{h} px")
        if lat: st.metric("Lat (DEMO)", f"{lat:.4f}°")
        if lon: st.metric("Lon (DEMO)", f"{lon:.4f}°")
        if depth_m: st.metric("Depth (DEMO)", f"{depth_m} m")

    # ── Pipeline run ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-hdr">3. Run AI Pipeline</div>', unsafe_allow_html=True)

    st.markdown(pipeline_bar(-1), unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("🚀 Run Full AI Pipeline", type="primary", use_container_width=True)

    if run_btn:
        pipeline = get_pipeline()

        # Animated pipeline stages
        stages = [
            "📥 SONAR DATA RECEIVED",
            "🔬 ANALYZING SONAR QUALITY",
            "⚙️ PREPROCESSING (CLAHE + DENOISE)",
            "🤖 RUNNING YOLOv8-NANO DETECTION",
            "🌑 ANALYZING ACOUSTIC SHADOW",
            "🧬 RUNNING ANOMALY DETECTION",
            "⚡ FUSING EVIDENCE SIGNALS",
            "📍 GENERATING GEO-TAGGED RESULT",
            "✅ ANALYSIS COMPLETE",
        ]

        status_box = st.empty()
        prog_box   = st.empty()

        t_start = time.time()
        for idx, stage_label in enumerate(stages[:-1]):
            status_box.markdown(
                f'<div class="fusion-result" style="margin:0.3rem 0;">'
                f'<span style="color:var(--accent);font-weight:700;font-size:0.88rem;">'
                f'▶ {stage_label}</span></div>',
                unsafe_allow_html=True
            )
            prog_box.markdown(pipeline_bar(idx), unsafe_allow_html=True)
            if idx == 2:
                # Real work happens here
                result = pipeline.run(
                    raw_img,
                    image_id=f"IMG-{int(time.time())}",
                    scenario_type=scenario_type,
                    annotations=annotations,
                    lat=lat, lon=lon, depth_m=depth_m,
                )
            time.sleep(0.18)

        status_box.markdown(
            '<div class="fusion-result" style="margin:0.3rem 0;">'
            '<span style="color:var(--success);font-weight:700;font-size:0.88rem;">'
            '✅ ANALYSIS COMPLETE</span></div>',
            unsafe_allow_html=True
        )
        prog_box.markdown(pipeline_bar(9), unsafe_allow_html=True)
        time.sleep(0.3)
        status_box.empty()

        st.session_state["last_result"] = result
        st.session_state["last_raw"]    = raw_img.copy()
        st.session_state["last_label"]  = img_label

        # Auto-save
        try:
            from services.mission_service import save_image_record, save_detection
            save_image_record("MISSION-001", result.image_id, img_label, img_label,
                              scenario_type or "uploaded", img_label,
                              result.quality.get("quality","UNKNOWN"),
                              result.quality.get("score", 0.0), lat, lon, depth_m)
            for det in result.detections:
                save_detection("MISSION-001", result.image_id, det, lat, lon, depth_m)
        except Exception:
            pass

    # ── Results ────────────────────────────────────────────────────────────────
    result = st.session_state.get("last_result")
    if result is None:
        st.info("👆 Click **Run Full AI Pipeline** to start the analysis.")
        return

    st.success(f"✅ Pipeline complete — {result.timing.get('total_ms',0):.0f} ms total")
    for w in result.warnings:
        st.warning(w)

    # ── Image triptych ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-hdr">4. Sonar Analysis Results</div>', unsafe_allow_html=True)

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        show_sonar(result.raw_image, "📥 RAW SONAR")
    with ic2:
        show_sonar(result.preprocessed_image, "⚙️ CLAHE ENHANCED")
    with ic3:
        show_sonar(result.annotated_image, "🎯 AI DETECTION RESULT")

    # ── Quality assessment ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-hdr">5. Image Quality Assessment</div>', unsafe_allow_html=True)
    q = result.quality
    qc1, qc2 = st.columns([1, 3])
    with qc1:
        q_icon = {"GOOD":"🟢","ACCEPTABLE":"🟡","POOR":"🔴","UNKNOWN":"⚪"}.get(q.get("quality","UNKNOWN"),"⚪")
        st.markdown(f"### {q_icon} {q.get('quality','UNKNOWN')}")
        st.caption(f"Quality score: {q.get('score',0):.2f} / 1.00")
    with qc2:
        m = q.get("metrics", {})
        qm = st.columns(5)
        qm[0].metric("Sharpness",  f"{m.get('sharpness',0):.1f}")
        qm[1].metric("Contrast",   f"{m.get('contrast',0):.1f}")
        qm[2].metric("Noise Est.", f"{m.get('noise_estimate',0):.1f}")
        qm[3].metric("Brightness", f"{m.get('mean_brightness',0):.1f}")
        qm[4].metric("Missing %",  f"{m.get('missing_data_pct',0):.1f}%")
        if q.get("reasons"):
            st.caption("⚠️ " + " | ".join(q["reasons"]))

    # ── Processing timing ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-hdr">6. Processing Time (MEASURED)</div>', unsafe_allow_html=True)
    tc = st.columns(5)
    tc[0].metric("Preprocessing", f"{result.timing.get('preprocess_ms',0):.0f} ms")
    tc[1].metric("Detection",     f"{result.timing.get('detection_ms',0):.0f} ms")
    tc[2].metric("Anomaly",       f"{result.timing.get('anomaly_ms',0):.0f} ms")
    tc[3].metric("Shadow",        f"{result.timing.get('shadow_ms',0):.0f} ms")
    tc[4].metric("TOTAL",         f"{result.timing.get('total_ms',0):.0f} ms")

    # ── Detections ─────────────────────────────────────────────────────────────
    st.markdown("---")
    nd = len(result.detections)
    st.markdown(f'<div class="sec-hdr">7. Detections ({nd} found)</div>', unsafe_allow_html=True)
    st.markdown(f"Detector: {mode_badge(result.mode)}", unsafe_allow_html=True)

    if nd == 0:
        st.info("No objects detected — normal seabed or below detection threshold.")
    else:
        for det in result.detections:
            sev = det.severity
            card_class = ("sonar-card sonar-card-anomaly" if det.is_anomaly
                          else f"sonar-card sonar-card-{sev.lower()}")
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

            dc1, dc2, dc3 = st.columns([2, 2, 1])
            with dc1:
                icon = "🔴 ANOMALY" if det.is_anomaly else "🔵 KNOWN OBJECT"
                st.markdown(f"**{icon} — {det.display_name}**")
                st.markdown(
                    f"Severity: {sev_badge(sev)}&nbsp;&nbsp;"
                    f"{mode_badge(det.mode)}&nbsp;&nbsp;"
                    f"ID: <code>{det.detection_id[:14]}</code>",
                    unsafe_allow_html=True
                )
                if not det.is_anomaly:
                    from services.pipeline_service import classify_natural_vs_artificial
                    nat = classify_natural_vs_artificial(det.class_name, det.confidence, det.anomaly_score)
                    ntype = "🏭 MAN-MADE" if nat["is_manmade"] else "🌊 NATURAL"
                    st.caption(f"{ntype} — {nat['reason']}")

            with dc2:
                bars_html = (
                    score_bar(det.confidence,    "#19C3D1", "Detection Confidence") +
                    score_bar(det.shadow_score,  "#F4C95D", "Shadow Consistency") +
                    score_bar(det.anomaly_score, "#B78CFF", "Anomaly Score") +
                    score_bar(det.evidence_score,"#38D39F", "Evidence Score")
                )
                st.markdown(bars_html, unsafe_allow_html=True)

            with dc3:
                st.metric("Width",  f"{det.bbox_width_px} px")
                st.metric("Height", f"{det.bbox_height_px} px")
                if det.lat:
                    st.caption(f"📍 {det.lat:.4f}°, {det.lon:.4f}°")
                    st.caption("⚠️ DEMO LOCATION")
                else:
                    st.caption("📍 Geolocation unavailable")

            # Shadow details expander
            sd = det.shadow_details
            if sd and sd.get("shadow_score", 0) > 0.05:
                with st.expander("🌑 Acoustic Shadow Details"):
                    sa, sb, sc_, sd_ = st.columns(4)
                    sa.metric("Shadow Score",   f"{sd.get('shadow_score',0):.0%}")
                    sb.metric("Shadow Area",    f"{sd.get('shadow_area_px',0):,} px")
                    sc_.metric("Obj/Shadow",    f"{sd.get('shadow_to_object_ratio',0):.2f}")
                    sd_.metric("Shadow Length", f"{sd.get('shadow_length_px',0)} px")
                    st.caption(f"🔭 *{sd.get('height_estimate','Unavailable')}*")
            else:
                st.caption("🌑 Acoustic shadow: no reliable shadow detected")

            st.markdown("</div>", unsafe_allow_html=True)

    # ── Evidence Fusion visualization ──────────────────────────────────────────
    if nd > 0:
        st.markdown("---")
        st.markdown('<div class="sec-hdr">8. Evidence Fusion</div>', unsafe_allow_html=True)
        det0 = result.detections[0]
        fc1,fc2,fc3,fc4,fc5 = st.columns([2,1,2,1,2])
        with fc1:
            is_loaded = det0.confidence > 0.3
            cls = "fusion-signal loaded" if is_loaded else "fusion-signal"
            st.markdown(f"""<div class="{cls}">
                🤖 AI DETECTION<br><b style="font-size:1.1rem">{det0.confidence:.0%}</b>
            </div>""", unsafe_allow_html=True)
        with fc2:
            st.markdown('<div style="text-align:center;color:#1B5263;font-size:1.4rem;padding-top:0.5rem">+</div>',
                        unsafe_allow_html=True)
        with fc3:
            is_loaded = det0.shadow_score > 0.1
            cls = "fusion-signal loaded" if is_loaded else "fusion-signal"
            st.markdown(f"""<div class="{cls}">
                🌑 SHADOW<br><b style="font-size:1.1rem">{det0.shadow_score:.0%}</b>
            </div>""", unsafe_allow_html=True)
        with fc4:
            st.markdown('<div style="text-align:center;color:#1B5263;font-size:1.4rem;padding-top:0.5rem">→</div>',
                        unsafe_allow_html=True)
        with fc5:
            st.markdown(f"""<div class="fusion-result">
                ⚡ EVIDENCE SCORE<br>
                <b style="font-size:1.6rem;color:#19C3D1">{det0.evidence_pct:.0f}%</b><br>
                <small style="color:#91B7C0">Severity: </small>{sev_badge(det0.severity)}
            </div>""", unsafe_allow_html=True)
        st.caption("Evidence Score = weighted sum of detector confidence, shadow consistency, "
                   "anomaly score, image quality. Engineering heuristic — not a calibrated accuracy metric.")

    # ── Anomaly section ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-hdr">9. Anomaly Analysis</div>', unsafe_allow_html=True)
    ar = result.anomaly_result
    anom_score = ar.get("anomaly_score", 0)
    is_anom = ar.get("is_anomalous", False)
    ac1, ac2 = st.columns([1, 3])
    with ac1:
        icon = "🔴 ANOMALOUS" if is_anom else "🟢 NORMAL"
        st.markdown(f"### {icon}")
        st.metric("Anomaly Score", f"{anom_score:.3f}")
        st.caption(f"Mode: {ar.get('mode','DEMO')}")
    with ac2:
        for r in ar.get("reasons", []):
            st.caption(f"• {r}")
        if not ar.get("reasons"):
            st.caption("No anomaly indicators detected.")
        st.caption("⚠️ Anomaly detector: PROTOTYPE statistical baseline (demo mode)")

    # ── Export section ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-hdr">10. Export Results</div>', unsafe_allow_html=True)
    ex1, ex2, ex3, ex4 = st.columns(4)

    with ex1:
        if result.annotated_image is not None:
            st.download_button("📥 Annotated Image",
                               img_bytes(result.annotated_image),
                               file_name=f"sonar_result_{result.image_id}.png",
                               mime="image/png", use_container_width=True)
    with ex2:
        det_json = json.dumps({
            "image_id": result.image_id,
            "data_label": "DEMO/SYNTHETIC",
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "detections": [d.to_dict() for d in result.detections],
        }, indent=2, default=str)
        st.download_button("📄 JSON Report",
                           det_json,
                           file_name=f"detections_{result.image_id}.json",
                           mime="application/json", use_container_width=True)
    with ex3:
        if result.detections:
            import csv as csvlib
            csv_buf = io.StringIO()
            fields = ["detection_id","class_name","confidence","anomaly_score",
                      "shadow_score","evidence_score","severity","lat","lon",
                      "depth_m","geo_label","mode"]
            w = csvlib.DictWriter(csv_buf, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows([d.to_dict() for d in result.detections])
            st.download_button("📊 CSV Report",
                               csv_buf.getvalue(),
                               file_name=f"detections_{result.image_id}.csv",
                               mime="text/csv", use_container_width=True)
    with ex4:
        if st.button("📑 Generate PDF Report", use_container_width=True, type="primary"):
            with st.spinner("Generating professional PDF report..."):
                try:
                    from utils.pdf_report import generate_pdf_report
                    pdf_bytes = generate_pdf_report(result)
                    st.download_button(
                        "⬇️ Download PDF",
                        pdf_bytes,
                        file_name=f"SIH26057_report_{result.image_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.success("PDF ready!")
                except Exception as e:
                    st.error(f"PDF generation error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Detections
# ══════════════════════════════════════════════════════════════════════════════

def page_detections():
    st.markdown("## 🎯 All Detections")
    ensure_db()
    from services.mission_service import get_all_detections, get_all_missions, update_operator_feedback

    missions = get_all_missions()
    opts = {"All Missions": None}
    opts.update({m["name"]: m["mission_id"] for m in missions})
    sel = st.selectbox("Filter by mission:", list(opts.keys()))
    mid = opts[sel]
    dets = get_all_detections(mission_id=mid, limit=500)

    if not dets:
        st.info("No detections found. Run demo setup or analyze images.")
        return

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        classes = ["All"] + sorted(set(d["class_name"] for d in dets if d.get("class_name")))
        sel_cl = st.selectbox("Class:", classes)
    with fc2:
        sel_sv = st.selectbox("Severity:", ["All","HIGH","MEDIUM","LOW","UNKNOWN"])
    with fc3:
        only_anom = st.checkbox("Anomalies only")

    filtered = dets
    if sel_cl != "All":    filtered = [d for d in filtered if d.get("class_name") == sel_cl]
    if sel_sv != "All":    filtered = [d for d in filtered if d.get("severity") == sel_sv]
    if only_anom:          filtered = [d for d in filtered if d.get("is_anomaly")]

    st.markdown(f"**{len(filtered)}** detection(s)")

    df = pd.DataFrame(filtered)
    if df.empty:
        st.info("No detections match filters."); return

    show_cols = ["detection_id","image_id","class_name","confidence","anomaly_score",
                 "evidence_score","severity","is_anomaly","operator_status","mode"]
    df_d = df[[c for c in show_cols if c in df.columns]].copy()
    for c in ["confidence","anomaly_score","evidence_score"]:
        if c in df_d:
            df_d[c] = df_d[c].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    st.dataframe(df_d, use_container_width=True, hide_index=True)

    # ── Human-in-the-loop ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-hdr">Human-in-the-Loop Review</div>', unsafe_allow_html=True)
    st.caption("Operator feedback stored for active learning pipeline")

    if filtered:
        sel_did = st.selectbox("Select detection to review:", [d["detection_id"] for d in filtered[:50]])
        sel_d = next((d for d in filtered if d["detection_id"] == sel_did), None)
        if sel_d:
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"**Class:** {sel_d.get('class_name','—')}")
                st.markdown(f"**Confidence:** {sel_d.get('confidence',0):.0%}")
                st.markdown(f"**Severity:** {sel_d.get('severity','—')}")
                st.markdown(f"**Current:** {sel_d.get('operator_status','pending')}")
            with rc2:
                action = st.selectbox("Action:", ["accepted","rejected","relabeled","uncertain"])
                new_lbl = st.text_input("New label (if relabeled):") if action == "relabeled" else None
                note = st.text_area("Operator note:", height=70)
                if st.button("✅ Submit", type="primary"):
                    update_operator_feedback(sel_did, action, new_lbl, note)
                    st.success("Feedback saved for active learning.")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Anomaly Analysis
# ══════════════════════════════════════════════════════════════════════════════

def page_anomalies():
    st.markdown("## ⚠️ Anomaly Analysis")
    ensure_db()
    from services.mission_service import get_all_detections
    dets = get_all_detections(limit=500)
    anoms = [d for d in dets if d.get("is_anomaly")]

    st.metric("Unknown Anomalies", len(anoms))
    st.caption("Objects that did not match any known debris class and triggered the anomaly detector.")
    st.caption("⚠️ PROTOTYPE MODE — statistical pixel baseline, not a trained neural network.")

    if not anoms:
        st.info("No anomalies detected yet. Try the 'unknown_anomaly' scenario in Sonar Analysis.")
        return

    # Anomaly visualization with purple theme
    for a in anoms:
        st.markdown(f"""<div class="sonar-card sonar-card-anomaly">
            <div style="color:#B78CFF;font-weight:700;font-size:0.9rem">⚠️ UNKNOWN ANOMALY</div>
            <div style="color:#91B7C0;font-size:0.78rem">ID: {a['detection_id']} | Image: {a['image_id']}</div>
            <div style="margin-top:0.4rem">
              {score_bar(a.get('anomaly_score',0),'#B78CFF','Anomaly Score')}
              {score_bar(a.get('evidence_score',0),'#38D39F','Evidence Score')}
            </div>
            <div style="margin-top:0.4rem;font-size:0.75rem;color:#F4C95D">▶ REVIEW REQUIRED</div>
        </div>""", unsafe_allow_html=True)

    df = pd.DataFrame(anoms)[["detection_id","image_id","anomaly_score","evidence_score","severity","mode"]]
    st.dataframe(df, use_container_width=True, hide_index=True)

    scores = [d.get("anomaly_score", 0) for d in anoms]
    fig = px.histogram(x=scores, nbins=10, title="Anomaly Score Distribution",
                       color_discrete_sequence=["#B78CFF"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#91B7C0")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Acoustic Shadow
# ══════════════════════════════════════════════════════════════════════════════

def page_shadow():
    st.markdown("## 🌑 Acoustic Shadow Analysis")
    st.caption("Side-scan sonar objects produce a bright return + dark acoustic shadow on the far side")

    ensure_db()
    from services.mission_service import get_all_detections
    dets = get_all_detections(limit=500)
    shadow_dets = [d for d in dets if d.get("shadow_score", 0) > 0.05]

    st.metric("Detections with Shadow Evidence", len(shadow_dets))

    # Physics explanation
    with st.expander("📖 Acoustic Shadow Physics"):
        st.markdown("""
**Side-Scan Sonar Shadow Formation:**
- The sonar emits acoustic pulses perpendicular to the vehicle track
- Objects facing the sonar create a **bright (high intensity) return**
- Objects block the sonar signal → **dark shadow** on the far side (away from nadir)
- Shadow length is related to object height (requires sonar geometry metadata)

**Shadow Score Components (prototype heuristic):**
1. Object brightness > threshold (30%)
2. Shadow darkness < threshold (30%)
3. Contrast ratio (20%)
4. Shadow on correct side (10%)
5. Shadow/object area ratio (10%)

**Height estimation requires:** sonar altitude, depression angle, slant range.
Without this metadata: *"Estimated height unavailable"*
        """)

    if not shadow_dets:
        st.info("No detections with shadow evidence found. Try the 'strong_shadow' demo scenario.")
        return

    for d in shadow_dets:
        score = d.get("shadow_score", 0)
        color = "#38D39F" if score > 0.6 else "#F4C95D" if score > 0.3 else "#91B7C0"
        st.markdown(f"""<div class="sonar-card">
            <div style="font-weight:700;color:#E8F7FA">{d.get('class_name','—')}</div>
            <div style="font-size:0.75rem;color:#91B7C0;margin-bottom:0.4rem">ID: {d['detection_id'][:14]}</div>
            <div style="font-size:0.85rem;color:{color};margin-bottom:0.3rem">
                Shadow Consistency: <b>{score:.0%}</b>
            </div>
            {'<div style="font-size:0.72rem;color:#91B7C0">Shadow region localized and measured</div>'
             if score > 0.3 else
             '<div style="font-size:0.72rem;color:#91B7C0">SHADOW NOT RELIABLE — insufficient evidence</div>'}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Map / Geolocation
# ══════════════════════════════════════════════════════════════════════════════

def page_map():
    st.markdown("## 🗺️ Geolocation")
    st.warning("⚠️ All coordinates shown are **DEMO / SIMULATED** — not real survey locations.")

    ensure_db()
    from services.mission_service import get_all_detections
    import folium
    from streamlit_folium import st_folium

    dets = get_all_detections(limit=500)
    geo = [d for d in dets if d.get("lat") and d.get("lon")]

    if not geo:
        st.info("No geo-tagged detections. Run demo setup to generate demo coordinates.")
        return

    fc1, fc2 = st.columns(2)
    with fc1:
        classes = ["All"] + sorted(set(d["class_name"] for d in geo if d.get("class_name")))
        sel_cl = st.selectbox("Class:", classes)
    with fc2:
        sel_sv = st.selectbox("Severity:", ["All","HIGH","MEDIUM","LOW","UNKNOWN"])

    filtered = geo
    if sel_cl != "All": filtered = [d for d in filtered if d.get("class_name") == sel_cl]
    if sel_sv != "All": filtered = [d for d in filtered if d.get("severity") == sel_sv]
    st.markdown(f"**{len(filtered)}** detection(s) on map")

    clat = np.mean([d["lat"] for d in filtered])
    clon = np.mean([d["lon"] for d in filtered])
    m = folium.Map(location=[clat, clon], zoom_start=16, tiles="CartoDB dark_matter")

    sev_map = {"HIGH":"red","MEDIUM":"orange","LOW":"green","UNKNOWN":"gray"}
    for d in filtered:
        color = sev_map.get(d.get("severity","UNKNOWN"), "blue")
        popup = f"""<div style="min-width:200px;font-family:sans-serif">
            <b>{d.get('class_name','—')}</b><br>
            ID: {d['detection_id'][:14]}<br>
            Confidence: {d.get('confidence',0):.0%}<br>
            Evidence: {d.get('evidence_score',0):.0%}<br>
            Severity: <b>{d.get('severity','—')}</b><br>
            Depth: {d.get('depth_m','—')} m (DEMO)<br>
            <hr><small>⚠️ DEMO / SIMULATED COORDINATES</small>
        </div>"""
        folium.Marker(
            location=[d["lat"], d["lon"]],
            popup=folium.Popup(popup, max_width=260),
            tooltip=f"{d.get('class_name','?')} | {d.get('severity','?')}",
            icon=folium.Icon(color=color, icon="info-sign", prefix="glyphicon"),
        ).add_to(m)

    st_folium(m, height=520, use_container_width=True)
    st.markdown("**Legend:** 🔴 HIGH &nbsp; 🟠 MEDIUM &nbsp; 🟢 LOW &nbsp; ⚫ UNKNOWN")
    st.caption("⚠️ DEMO / SIMULATED COORDINATES — Not real survey locations.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Analytics
# ══════════════════════════════════════════════════════════════════════════════

def page_analytics():
    st.markdown("## 📊 Analytics")
    ensure_db()
    from services.mission_service import get_statistics, get_all_detections
    stats = get_statistics()
    dets = get_all_detections(limit=1000)

    kc = st.columns(4)
    kc[0].metric("Total Detections", stats["total_detections"])
    kc[1].metric("Anomaly Rate",  f"{stats['total_anomalies']/max(stats['total_detections'],1)*100:.1f}%")
    kc[2].metric("High Risk %",   f"{stats['high_risk']/max(stats['total_detections'],1)*100:.1f}%")
    kc[3].metric("Avg Evidence",  f"{stats['avg_evidence_pct']:.1f}%")

    if not dets:
        st.info("No detections yet."); return
    df = pd.DataFrame(dets)
    clo = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
               font=dict(color="#91B7C0"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">Confidence Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="confidence", nbins=20, color_discrete_sequence=["#19C3D1"])
        fig.update_layout(**clo); st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="sec-hdr">Evidence vs. Confidence</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x="confidence", y="evidence_score", color="severity",
                         color_discrete_map={"HIGH":"#FF667A","MEDIUM":"#F4C95D",
                                             "LOW":"#38D39F","UNKNOWN":"#91B7C0"},
                         hover_data=["class_name"])
        fig.update_layout(**clo); st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">Model Performance</div>', unsafe_allow_html=True)
    st.warning(
        "Precision / Recall / F1 / mAP are **NOT available** — "
        "YOLOv8-Nano has not been trained on a labelled sonar dataset yet. "
        "Results shown are from DEMO / PROTOTYPE mode."
    )
    st.markdown("""
| Metric | Status |
|---|---|
| Precision | NOT MEASURED — training pending |
| Recall | NOT MEASURED — training pending |
| F1 Score | NOT MEASURED — training pending |
| mAP@50 | NOT MEASURED — training pending |
| mAP@50:95 | NOT MEASURED — training pending |

*These will be populated after training on the SubPipe dataset or annotated marine-debris data.*
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Missions
# ══════════════════════════════════════════════════════════════════════════════

def page_missions():
    st.markdown("## 🚢 Missions")
    ensure_db()
    from services.mission_service import get_all_missions, get_mission_images
    missions = get_all_missions()
    if not missions:
        st.info("No missions. Click 'Run Demo Setup' in sidebar."); return
    for m in missions:
        with st.expander(f"🚢 {m['name']} | {m['mission_id']} | {m['image_count']} images | {m['detection_count']} detections"):
            mc = st.columns(3)
            mc[0].metric("Images", m["image_count"])
            mc[1].metric("Detections", m["detection_count"])
            mc[2].metric("Anomalies", m["anomaly_count"])
            st.caption(f"Date: {m['date']} | Area: {m['area']}")
            st.caption(f"Label: {m['data_label']}")
            imgs = get_mission_images(m["mission_id"])
            if imgs:
                df = pd.DataFrame(imgs)[["image_id","filename","scenario_type","quality_label","processed"]]
                st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: System Status
# ══════════════════════════════════════════════════════════════════════════════

def page_models():
    st.markdown("## System Status")

    # ── Model Status Card ─────────────────────────────────────────────────────
    pipeline = get_pipeline()
    mode = pipeline.detector.mode
    model_path = getattr(pipeline.detector, 'model_path', 'models/best.pt')
    model_exists = Path(model_path).exists()

    if mode == "REAL":
        st.markdown("""
        <div class="sonar-card" style="border-left:4px solid #38D39F;">
          <div style="font-size:1.1rem;font-weight:700;color:#38D39F">MODEL READY</div>
          <div style="color:#91B7C0;font-size:0.8rem;margin-top:0.3rem">YOLOv8-Nano loaded — REAL INFERENCE ENABLED</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sonar-card" style="border-left:4px solid #F4C95D;">
          <div style="font-size:1.1rem;font-weight:700;color:#F4C95D">MODEL NOT INSTALLED</div>
          <div style="color:#91B7C0;font-size:0.8rem;margin-top:0.3rem">Using DEMO/SYNTHETIC baseline — Training required</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Detection Model</div>', unsafe_allow_html=True)
    mc = st.columns(4)
    mc[0].metric("Mode", mode)
    mc[1].metric("Architecture", "YOLOv8-Nano")
    mc[2].metric("Model File", "Found" if model_exists else "Not Found")
    mc[3].metric("Expected Path", "models/best.pt")

    if mode == "DEMO":
        st.warning(
            "**DEMO MODE ACTIVE** — `models/best.pt` not found.\n\n"
            "The application is using a rule-based demo detector.\n\n"
            "**Detections shown are NOT from a trained YOLO model.**\n\n"
            "To activate REAL mode:\n"
            "1. `python train.py --data <path/to/data.yaml> --epochs 100`\n"
            "2. Best weights auto-copied to `models/best.pt`\n"
            "3. Restart app — auto-switches to REAL INFERENCE mode"
        )

    st.markdown("---")

    # ── Evaluation status ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Model Evaluation</div>', unsafe_allow_html=True)
    eval_path = Path("reports/evaluation_results.json")
    if eval_path.exists():
        try:
            import json as _json
            with open(eval_path, encoding="utf-8") as f:
                eval_data = _json.load(f)
            m = eval_data.get("metrics", {})
            ec = st.columns(5)
            ec[0].metric("Precision", f"{m.get('precision',0):.3f}")
            ec[1].metric("Recall",    f"{m.get('recall',0):.3f}")
            ec[2].metric("F1",        f"{m.get('f1',0):.3f}" if m.get('f1') else "N/A")
            ec[3].metric("mAP@50",    f"{m.get('map50',0):.3f}")
            ec[4].metric("mAP@50:95", f"{m.get('map50_95',0):.3f}")
            st.caption(f"Evaluated on: {eval_data.get('dataset','?')} | Split: {eval_data.get('split','?')} | {eval_data.get('timestamp','?')[:10]}")
            st.caption(eval_data.get('note', ''))
        except Exception as e:
            st.error(f"Could not read evaluation results: {e}")
    else:
        st.info("Evaluation unavailable — trained model required.\n\nAfter training: `python evaluate.py`")

    st.markdown("---")
    st.markdown('<div class="sec-hdr">Dataset Adapter</div>', unsafe_allow_html=True)

    try:
        from datasets.adapter import DatasetAdapter
        adapter = DatasetAdapter()
        info = adapter.inspect()
        dc = st.columns(3)
        dc[0].metric("Dataset Mode", info.get("mode", "?").upper())
        dc[1].metric("Total Images", info.get("total_images", 0))
        dc[2].metric("Data Label",   info.get("data_label", "?"))
        st.caption(f"Path: {info.get('dataset_path','N/A')}")
        if info.get("classes"):
            st.markdown("**Classes (from actual annotation files):**")
            for cls, cnt in info["classes"].items():
                st.markdown(f"- `{cls}`: {cnt} annotations")
        st.caption(info.get("note", ""))
    except Exception as e:
        st.error(f"Dataset adapter error: {e}")

    st.markdown("""
---
**SubPipe Key Facts (from official docs — NOT guessed):**
- 1 class: **`Pipeline`** — NOT marine debris
- HF: 5,030 images (5000x500 px) / 3,172 annotations
- LF: 5,000 images (2500x500 px) / 3,163 annotations
- Split: **by mission chunk** (prevent temporal leakage from sequential frames)

```bash
# Inspect actual class names before training:
python train.py --inspect-only --data D:/SubPipe/data.yaml

# Train:
python train.py --data D:/SubPipe/data.yaml --epochs 100

# Evaluate:
python evaluate.py --model models/best.pt --data D:/SubPipe/data.yaml
```
    """)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">Anomaly Detection Module</div>', unsafe_allow_html=True)
    st.warning(
        "**Method: Statistical pixel-intensity baseline (PROTOTYPE)**\n\n"
        "Compares candidate region intensity, texture entropy, and edge density against background statistics.\n\n"
        "This is NOT a trained neural network.\n\n"
        "Future: replace with trained convolutional autoencoder."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Reports
# ══════════════════════════════════════════════════════════════════════════════

def page_reports():
    st.markdown("## 📋 Reports & Export")
    ensure_db()
    from services.mission_service import get_all_missions, export_detections_csv, export_detections_json

    missions = get_all_missions()
    opts = {"All Missions": None}
    opts.update({m["name"]: m["mission_id"] for m in missions})
    sel = st.selectbox("Mission:", list(opts.keys()))
    mid = opts[sel]

    st.markdown('<div class="sec-hdr">Export Detections</div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    with ec1:
        csv_data = export_detections_csv(mission_id=mid)
        st.download_button("📊 Download CSV", csv_data,
                           "sih26057_detections.csv", "text/csv", use_container_width=True)
    with ec2:
        json_data = export_detections_json(mission_id=mid)
        st.download_button("📄 Download JSON", json_data,
                           "sih26057_detections.json", "application/json", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">PDF Report — Last Analysis</div>', unsafe_allow_html=True)
    result = st.session_state.get("last_result")
    if result is None:
        st.info("No analysis result in session. Go to **Sonar Analysis** and run the pipeline first.")
    else:
        if st.button("📑 Generate PDF Report", type="primary", use_container_width=False):
            with st.spinner("Generating professional PDF report..."):
                try:
                    from utils.pdf_report import generate_pdf_report
                    pdf_bytes = generate_pdf_report(result, mission_id=mid or "MISSION-001")
                    st.download_button("⬇️ Download PDF Report",
                                       pdf_bytes,
                                       f"SIH26057_report_{result.image_id}.pdf",
                                       "application/pdf")
                    st.success("PDF generated successfully!")
                except Exception as e:
                    st.error(f"PDF error: {e}")

    st.markdown("---")
    st.markdown('<div class="sec-hdr">SubPipe Dataset Report</div>', unsafe_allow_html=True)
    rpt_path = Path("reports/subpipe_dataset_report.md")
    if rpt_path.exists():
        with open(rpt_path, encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Dataset report not generated yet.")
        if st.button("Generate SubPipe Report"):
            import sys as _sys
            old = _sys.argv
            _sys.argv = ["inspect_subpipe.py"]
            try:
                from scripts.inspect_subpipe import main as run_inspect
                run_inspect()
                st.success("Report generated: reports/subpipe_dataset_report.md")
                st.rerun()
            finally:
                _sys.argv = old


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    render_header()
    render_sidebar()

    page = st.session_state.get("page", "dashboard")
    dispatch = {
        "dashboard":  page_dashboard,
        "analysis":   page_sonar_analysis,
        "detections": page_detections,
        "anomalies":  page_anomalies,
        "shadow":     page_shadow,
        "map":        page_map,
        "analytics":  page_analytics,
        "missions":   page_missions,
        "models":     page_models,
        "reports":    page_reports,
    }
    dispatch.get(page, page_dashboard)()

if __name__ == "__main__":
    main()
