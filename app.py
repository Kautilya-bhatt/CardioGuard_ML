"""
CardioGuard AI — Intelligent Heart Disease Prediction System
------------------------------------------------------------
Streamlit front-end for the Logistic Regression pipeline trained in
CardioGuard_ML.ipynb  (StandardScaler -> LogisticRegression).

Run:
    pip install -r requirements.txt
    streamlit run app.py

Keep these files in the same folder as app.py:
    Heart_disease.pki   (trained model, saved with joblib)   <- required
    train.csv           (optional, unlocks the Model Insights tab)
"""

from __future__ import annotations

import io
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODEL_CANDIDATES = [
    "Heart_disease.pki",
    "Heart_disease__1_.pki",
    "Heart_disease.pkl",
    "Heart_disease.joblib",
]
DATA_CANDIDATES = ["train.csv"]

FEATURES = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
    "EKG results", "Max HR", "Exercise angina", "ST depression",
    "Slope of ST", "Number of vessels fluro",
]
TARGET = "Heart Disease"

# min / max / dataset-mean taken from train.csv (used for radar + defaults)
RANGES = {
    "Age": (29, 77, 54.1),
    "BP": (94, 200, 130.5),
    "Cholesterol": (126, 564, 245.0),
    "Max HR": (71, 202, 152.8),
    "ST depression": (0.0, 6.2, 0.72),
    "Number of vessels fluro": (0, 3, 0.45),
}

CP_LABELS = {1: "Typical angina", 2: "Atypical angina",
             3: "Non-anginal pain", 4: "Asymptomatic"}
EKG_LABELS = {0: "Normal", 1: "ST-T wave abnormality", 2: "LV hypertrophy"}
SLOPE_LABELS = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
SEX_LABELS = {0: "Female", 1: "Male"}
YN_LABELS = {0: "No", 1: "Yes"}

PRESETS = {
    "low": dict(age=38, sex=0, cp=2, bp=118, chol=190, fbs=0, ekg=0,
                hr=172, angina=0, st=0.0, slope=1, vessels=0),
    "mid": dict(age=55, sex=1, cp=3, bp=135, chol=245, fbs=0, ekg=1,
                hr=150, angina=0, st=1.0, slope=2, vessels=1),
    "high": dict(age=63, sex=1, cp=4, bp=160, chol=300, fbs=1, ekg=2,
                 hr=118, angina=1, st=3.2, slope=2, vessels=3),
}

# ──────────────────────────────────────────────────────────────────────────
# 2. STYLING
# ──────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(1200px 600px at 8% -10%, rgba(0,217,255,.14), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(138,43,226,.16), transparent 55%),
        radial-gradient(800px 600px at 50% 120%, rgba(255,64,110,.10), transparent 60%),
        #070b14;
}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.4rem; max-width: 1350px;}

/* ── hero ── */
.hero {
    position: relative; overflow: hidden;
    padding: 30px 34px; border-radius: 24px;
    background: linear-gradient(135deg, rgba(0,217,255,.10), rgba(138,43,226,.12));
    border: 1px solid rgba(0,217,255,.22);
    box-shadow: 0 10px 50px rgba(0,217,255,.08);
    margin-bottom: 18px;
}
.hero h1 {
    margin: 0; font-size: 2.5rem; font-weight: 800; letter-spacing: -.5px;
    background: linear-gradient(90deg,#00d9ff,#7c5cff 55%,#ff4d8d);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {margin: 6px 0 0 0; color: #9fb3c8; font-size: 1.02rem;}
.pulse {
    display:inline-block; width:10px; height:10px; border-radius:50%;
    background:#00e5a0; margin-right:8px; box-shadow:0 0 0 0 rgba(0,229,160,.7);
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% {box-shadow:0 0 0 0 rgba(0,229,160,.6);}
    70% {box-shadow:0 0 0 12px rgba(0,229,160,0);}
    100% {box-shadow:0 0 0 0 rgba(0,229,160,0);}
}
.heartbeat {display:inline-block; animation: beat 1.2s infinite; transform-origin:center;}
@keyframes beat {0%,100%{transform:scale(1);} 15%{transform:scale(1.18);} 30%{transform:scale(1);} 45%{transform:scale(1.12);}}

/* ── glass cards ── */
.glass {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 18px; padding: 18px 20px;
    backdrop-filter: blur(10px);
    transition: transform .2s ease, border-color .2s ease;
    height: 100%;
}
.glass:hover {transform: translateY(-3px); border-color: rgba(0,217,255,.35);}
.kpi-label {color:#8aa0b8; font-size:.74rem; text-transform:uppercase; letter-spacing:1.2px; font-weight:600;}
.kpi-value {font-size:1.75rem; font-weight:800; color:#fff; line-height:1.2; margin-top:4px;}
.kpi-sub {color:#7f94ab; font-size:.8rem; margin-top:2px;}

/* ── verdict banner ── */
.verdict {
    border-radius: 20px; padding: 22px 26px; margin-bottom: 14px;
    border: 1px solid; position: relative; overflow: hidden;
}
.verdict h2 {margin:0; font-size:1.6rem; font-weight:800;}
.verdict p {margin:6px 0 0 0; color:#c8d4e2;}
.v-low  {background: linear-gradient(120deg, rgba(0,229,160,.16), rgba(0,229,160,.04)); border-color: rgba(0,229,160,.45);}
.v-mod  {background: linear-gradient(120deg, rgba(255,184,0,.16), rgba(255,184,0,.04)); border-color: rgba(255,184,0,.45);}
.v-high {background: linear-gradient(120deg, rgba(255,64,110,.18), rgba(255,64,110,.04)); border-color: rgba(255,64,110,.5);}

.chip {
    display:inline-block; padding:4px 12px; border-radius:999px; font-size:.78rem;
    font-weight:600; margin:3px 6px 3px 0; border:1px solid;
}
.chip-ok   {color:#00e5a0; border-color:rgba(0,229,160,.4); background:rgba(0,229,160,.08);}
.chip-warn {color:#ffb800; border-color:rgba(255,184,0,.4); background:rgba(255,184,0,.08);}
.chip-bad  {color:#ff4d7a; border-color:rgba(255,64,110,.45); background:rgba(255,64,110,.09);}

.section-title {font-size:1.15rem; font-weight:700; color:#e8f1fb; margin: 6px 0 10px 0;}
.disclaimer {
    margin-top: 22px; padding: 14px 18px; border-radius: 14px; font-size:.85rem;
    color:#c7b98a; background: rgba(255,184,0,.06); border:1px solid rgba(255,184,0,.25);
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1020, #0c1428);
    border-right: 1px solid rgba(0,217,255,.12);
}
section[data-testid="stSidebar"] .stMarkdown h3 {color:#00d9ff; font-size:.95rem; letter-spacing:.5px;}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {gap: 6px; background: rgba(255,255,255,.03); padding: 6px; border-radius: 14px;}
.stTabs [data-baseweb="tab"] {border-radius: 10px; padding: 8px 16px; font-weight:600; color:#8aa0b8;}
.stTabs [aria-selected="true"] {background: linear-gradient(90deg, rgba(0,217,255,.18), rgba(124,92,255,.18)); color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}

/* ── buttons ── */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; font-weight: 600; border: 1px solid rgba(0,217,255,.35);
    background: linear-gradient(90deg, rgba(0,217,255,.15), rgba(124,92,255,.15));
    color: #eaf6ff; transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color:#00d9ff; box-shadow: 0 0 18px rgba(0,217,255,.3); transform: translateY(-1px);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOT_BG = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#cfe0f1"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ──────────────────────────────────────────────────────────────────────────
# 3. LOADERS (cached)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CardioGuard model…")
def load_model():
    for name in MODEL_CANDIDATES:
        p = BASE_DIR / name
        if p.exists():
            return joblib.load(p), name
    return None, None


@st.cache_data(show_spinner="Reproducing the notebook's test split…")
def load_insights():
    """Rebuild the exact 75/25 stratified split (random_state=42) used in the
    notebook so that metrics shown here are true hold-out metrics."""
    path = next((BASE_DIR / n for n in DATA_CANDIDATES if (BASE_DIR / n).exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path).drop_duplicates()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    corr_df = df[FEATURES + ["Thallium"]].copy()
    corr_df["Target"] = (df[TARGET] == "Presence").astype(int)
    return dict(
        X_test=X_te, y_test=y_te,
        corr=corr_df.sample(min(len(corr_df), 60000), random_state=1).corr(),
        class_counts=df[TARGET].value_counts(),
        n_rows=len(df),
        age_by_class=df[["Age", TARGET]].sample(min(len(df), 40000), random_state=1),
    )


model, model_file = load_model()
if model is None:
    st.error(
        "❌ Model file not found. Put **Heart_disease.pki** (exported from your "
        "notebook) in the same folder as `app.py` and refresh."
    )
    st.stop()

scaler = model.named_steps["scaler"]
clf = model.named_steps["classifier"]
POS_IDX = list(model.classes_).index("Presence")

# ──────────────────────────────────────────────────────────────────────────
# 4. HELPERS
# ──────────────────────────────────────────────────────────────────────────
def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, POS_IDX]


def risk_band(p: float):
    if p < 0.30:
        return "Low Risk", "v-low", "#00e5a0", "🟢"
    if p < 0.60:
        return "Moderate Risk", "v-mod", "#ffb800", "🟡"
    return "High Risk", "v-high", "#ff4d7a", "🔴"


def contributions(row: pd.DataFrame) -> pd.DataFrame:
    """Exact per-feature contribution to the log-odds of 'Presence'
    relative to an average patient:  coef_i * (x_i - mean_i) / scale_i"""
    z = (row[FEATURES].values[0] - scaler.mean_) / scaler.scale_
    coef = clf.coef_[0] if POS_IDX == 1 else -clf.coef_[0]
    out = pd.DataFrame({"Feature": FEATURES, "Value": row[FEATURES].values[0],
                        "Contribution": coef * z})
    return out.sort_values("Contribution")


def gauge(p: float) -> go.Figure:
    _, _, color, _ = risk_band(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 54, "color": color}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#5d7189"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(0,229,160,.16)"},
                {"range": [30, 60], "color": "rgba(255,184,0,.16)"},
                {"range": [60, 100], "color": "rgba(255,64,110,.18)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 3}, "thickness": .8, "value": p * 100},
        },
    ))
    fig.update_layout(height=290, **{**PLOT_BG, "margin": dict(l=20, r=20, t=20, b=0)})
    return fig


def radar(row: pd.DataFrame) -> go.Figure:
    keys = list(RANGES.keys())
    def norm(k, v):
        lo, hi, _ = RANGES[k]
        return float(np.clip((v - lo) / (hi - lo), 0, 1))
    patient = [norm(k, row[k].iloc[0]) for k in keys]
    avg = [norm(k, RANGES[k][2]) for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=avg + avg[:1], theta=keys + keys[:1], name="Dataset average",
                                  line=dict(color="#7c5cff", dash="dot"),
                                  fill="toself", fillcolor="rgba(124,92,255,.12)"))
    fig.add_trace(go.Scatterpolar(r=patient + patient[:1], theta=keys + keys[:1], name="This patient",
                                  line=dict(color="#00d9ff", width=3),
                                  fill="toself", fillcolor="rgba(0,217,255,.18)"))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                                   gridcolor="rgba(255,255,255,.1)"),
                   angularaxis=dict(gridcolor="rgba(255,255,255,.1)")),
        legend=dict(orientation="h", y=-0.1), height=380, **PLOT_BG)
    return fig


def vital_flags(r: pd.Series):
    """Informational rule-of-thumb flags (NOT part of the ML model)."""
    flags = []
    bp = r["BP"]
    flags.append(("Blood pressure", f"{bp:.0f} mmHg",
                  "ok" if bp < 130 else "warn" if bp < 140 else "bad"))
    ch = r["Cholesterol"]
    flags.append(("Cholesterol", f"{ch:.0f} mg/dL",
                  "ok" if ch < 200 else "warn" if ch < 240 else "bad"))
    maxhr_pred = 220 - r["Age"]
    pct = r["Max HR"] / maxhr_pred * 100
    flags.append(("Max HR vs age-predicted", f"{pct:.0f}%",
                  "ok" if pct >= 85 else "warn" if pct >= 70 else "bad"))
    st_d = r["ST depression"]
    flags.append(("ST depression", f"{st_d:.1f} mm",
                  "ok" if st_d < 1 else "warn" if st_d < 2 else "bad"))
    flags.append(("Fasting blood sugar", ">120 mg/dL" if r["FBS over 120"] else "≤120 mg/dL",
                  "bad" if r["FBS over 120"] else "ok"))
    flags.append(("Exercise angina", "Present" if r["Exercise angina"] else "Absent",
                  "bad" if r["Exercise angina"] else "ok"))
    return flags


def build_report(r: pd.Series, p: float) -> str:
    label, *_ = risk_band(p)
    lines = [
        "CARDIOGUARD AI — PATIENT RISK REPORT",
        "=" * 44,
        f"Generated : {datetime.now():%d %b %Y, %H:%M}",
        f"Result    : {label}",
        f"Probability of heart disease : {p*100:.1f}%",
        "", "INPUT PARAMETERS", "-" * 44,
        f"Age                      : {r['Age']:.0f}",
        f"Sex                      : {SEX_LABELS[int(r['Sex'])]}",
        f"Chest pain type          : {CP_LABELS[int(r['Chest pain type'])]}",
        f"Resting BP               : {r['BP']:.0f} mmHg",
        f"Cholesterol              : {r['Cholesterol']:.0f} mg/dL",
        f"Fasting sugar > 120      : {YN_LABELS[int(r['FBS over 120'])]}",
        f"Resting EKG              : {EKG_LABELS[int(r['EKG results'])]}",
        f"Max heart rate           : {r['Max HR']:.0f} bpm",
        f"Exercise angina          : {YN_LABELS[int(r['Exercise angina'])]}",
        f"ST depression            : {r['ST depression']:.1f}",
        f"Slope of ST              : {SLOPE_LABELS[int(r['Slope of ST'])]}",
        f"Vessels (fluoroscopy)    : {int(r['Number of vessels fluro'])}",
        "", "DISCLAIMER",
        "Educational ML project (Logistic Regression). Not a medical diagnosis.",
        "Consult a qualified doctor for clinical decisions.",
    ]
    return "\n".join(lines)


def kpi(label, value, sub=""):
    st.markdown(
        f"<div class='glass'><div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR — PATIENT INPUT
# ──────────────────────────────────────────────────────────────────────────
DEFAULTS = PRESETS["mid"]
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:6px 0 2px'>"
        "<span class='heartbeat' style='font-size:2.6rem'>🫀</span>"
        "<div style='font-weight:800;font-size:1.3rem;color:#00d9ff'>CardioGuard AI</div>"
        "<div style='color:#7f94ab;font-size:.8rem'>Patient Parameters</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("###### ⚡ Quick presets")
    c1, c2, c3 = st.columns(3)
    c1.button("Low", on_click=apply_preset, args=("low",), width="stretch")
    c2.button("Mid", on_click=apply_preset, args=("mid",), width="stretch")
    c3.button("High", on_click=apply_preset, args=("high",), width="stretch")

    st.markdown("### 👤 Demographics")
    age = st.slider("Age (years)", 29, 77, key="age")
    sex = st.radio("Sex", [0, 1], format_func=SEX_LABELS.get, horizontal=True, key="sex")

    st.markdown("### 🩸 Vitals & Labs")
    bp = st.slider("Resting BP (mmHg)", 94, 200, key="bp")
    chol = st.slider("Cholesterol (mg/dL)", 126, 564, key="chol")
    fbs = st.radio("Fasting sugar > 120 mg/dL", [0, 1], format_func=YN_LABELS.get,
                   horizontal=True, key="fbs")

    st.markdown("### 🫀 Cardiac Tests")
    cp = st.selectbox("Chest pain type", list(CP_LABELS), format_func=CP_LABELS.get, key="cp")
    ekg = st.selectbox("Resting EKG", list(EKG_LABELS), format_func=EKG_LABELS.get, key="ekg")
    hr = st.slider("Max heart rate (bpm)", 71, 202, key="hr")
    angina = st.radio("Exercise-induced angina", [0, 1], format_func=YN_LABELS.get,
                      horizontal=True, key="angina")
    st_dep = st.slider("ST depression", 0.0, 6.2, step=0.1, key="st")
    slope = st.selectbox("Slope of ST segment", list(SLOPE_LABELS),
                         format_func=SLOPE_LABELS.get, key="slope")
    vessels = st.select_slider("Vessels coloured by fluoroscopy", [0, 1, 2, 3], key="vessels")

    st.caption(f"Model file: `{model_file}`")

patient = pd.DataFrame([{
    "Age": age, "Sex": sex, "Chest pain type": cp, "BP": bp, "Cholesterol": chol,
    "FBS over 120": fbs, "EKG results": ekg, "Max HR": hr, "Exercise angina": angina,
    "ST depression": st_dep, "Slope of ST": slope, "Number of vessels fluro": vessels,
}])
prob = float(predict_proba(patient)[0])
label, vclass, vcolor, emoji = risk_band(prob)

# ──────────────────────────────────────────────────────────────────────────
# 6. HERO
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='hero'><h1><span class='heartbeat'>🫀</span> CardioGuard AI</h1>"
    "<p><span class='pulse'></span>Intelligent Heart Disease Prediction System · "
    "Predict Early. Protect Every Beat.</p></div>",
    unsafe_allow_html=True,
)

tab_risk, tab_xai, tab_whatif, tab_batch, tab_insight, tab_about = st.tabs(
    ["🩺 Risk Assessment", "🔬 Explainability", "🧪 What-If Lab",
     "📂 Batch Prediction", "📊 Model Insights", "ℹ️ About"]
)

# ══════════════════ TAB 1 — RISK ASSESSMENT ══════════════════
with tab_risk:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.plotly_chart(gauge(prob), width="stretch", config={"displayModeBar": False})
    with right:
        msg = {
            "v-low": "Parameters look favourable. Maintain a healthy lifestyle and routine check-ups.",
            "v-mod": "Some indicators deserve attention. Consider a consultation and lifestyle review.",
            "v-high": "Multiple indicators are associated with heart disease. A clinical evaluation is advised.",
        }[vclass]
        st.markdown(
            f"<div class='verdict {vclass}'><h2>{emoji} {label}</h2>"
            f"<p>{msg}</p></div>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1: kpi("Presence", f"{prob*100:.1f}%", "heart disease")
        with k2: kpi("Absence", f"{(1-prob)*100:.1f}%", "healthy")
        with k3:
            odds = prob / max(1 - prob, 1e-9)
            kpi("Odds", f"{odds:.2f}", "presence : absence")
        st.write("")
        st.download_button(
            "⬇️ Download patient report",
            data=build_report(patient.iloc[0], prob),
            file_name=f"cardioguard_report_{datetime.now():%Y%m%d_%H%M}.txt",
            mime="text/plain", width="stretch",
        )

    st.markdown("<div class='section-title'>🧭 Clinical snapshot</div>", unsafe_allow_html=True)
    chips = "".join(
        f"<span class='chip chip-{lvl}'>{name}: {val}</span>"
        for name, val, lvl in vital_flags(patient.iloc[0])
    )
    st.markdown(chips, unsafe_allow_html=True)
    st.caption("Chips use common rule-of-thumb thresholds for context only — they are not inputs to the ML model.")

    st.markdown("<div class='section-title'>🕸️ Patient vs dataset average</div>", unsafe_allow_html=True)
    st.plotly_chart(radar(patient), width="stretch", config={"displayModeBar": False})

# ══════════════════ TAB 2 — EXPLAINABILITY ══════════════════
with tab_xai:
    st.markdown("<div class='section-title'>Why did the model say this?</div>", unsafe_allow_html=True)
    st.caption("Each bar is the exact push this feature gives to the log-odds of heart disease, "
               "compared with an average patient (coefficient × standardised value).")
    contrib = contributions(patient)
    colors = ["#ff4d7a" if v > 0 else "#00e5a0" for v in contrib["Contribution"]]
    fig = go.Figure(go.Bar(
        x=contrib["Contribution"], y=contrib["Feature"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.2f}" for v in contrib["Contribution"]], textposition="outside",
        customdata=contrib["Value"],
        hovertemplate="<b>%{y}</b><br>Value: %{customdata}<br>Contribution: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(height=470, xaxis_title="← lowers risk      raises risk →",
                      **PLOT_BG)
    fig.add_vline(x=0, line_color="rgba(255,255,255,.35)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    up = contrib[contrib["Contribution"] > 0].sort_values("Contribution", ascending=False).head(3)
    dn = contrib[contrib["Contribution"] < 0].head(3)
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='glass'><div class='kpi-label'>🔺 Top risk drivers</div>",
                    unsafe_allow_html=True)
        for _, r in up.iterrows():
            st.markdown(f"**{r['Feature']}** — `{r['Value']}` · {r['Contribution']:+.2f}")
        if up.empty: st.write("None")
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='glass'><div class='kpi-label'>🔻 Top protective factors</div>",
                    unsafe_allow_html=True)
        for _, r in dn.iterrows():
            st.markdown(f"**{r['Feature']}** — `{r['Value']}` · {r['Contribution']:+.2f}")
        if dn.empty: st.write("None")
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════ TAB 3 — WHAT-IF LAB ══════════════════
with tab_whatif:
    st.markdown("<div class='section-title'>Change one parameter, watch the risk move</div>",
                unsafe_allow_html=True)
    sweep_defs = {
        "Age": np.arange(29, 78, 1),
        "BP": np.arange(94, 201, 2),
        "Cholesterol": np.arange(126, 565, 6),
        "Max HR": np.arange(71, 203, 2),
        "ST depression": np.round(np.arange(0, 6.3, 0.1), 1),
    }
    feat = st.selectbox("Parameter to vary", list(sweep_defs))
    grid = sweep_defs[feat]
    sim = pd.concat([patient] * len(grid), ignore_index=True)
    sim[feat] = grid
    probs = predict_proba(sim) * 100
    cur = float(patient[feat].iloc[0])

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,229,160,.07)", line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="rgba(255,184,0,.07)", line_width=0)
    fig.add_hrect(y0=60, y1=100, fillcolor="rgba(255,64,110,.08)", line_width=0)
    fig.add_trace(go.Scatter(x=grid, y=probs, mode="lines", name="Risk",
                             line=dict(color="#00d9ff", width=4, shape="spline"),
                             fill="tozeroy", fillcolor="rgba(0,217,255,.08)"))
    fig.add_trace(go.Scatter(x=[cur], y=[prob * 100], mode="markers", name="Current",
                             marker=dict(size=15, color=vcolor, line=dict(color="#fff", width=2))))
    fig.update_layout(height=430, xaxis_title=feat, yaxis_title="Heart-disease probability (%)",
                      yaxis=dict(range=[0, 100]), **PLOT_BG)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    delta = probs[-1] - probs[0]
    st.info(f"Across the full {feat} range (other inputs fixed) the predicted risk changes by "
            f"**{delta:+.1f} percentage points**.")

# ══════════════════ TAB 4 — BATCH PREDICTION ══════════════════
with tab_batch:
    st.markdown("<div class='section-title'>Score many patients at once</div>", unsafe_allow_html=True)
    template = pd.DataFrame([PRESETS_ROW for PRESETS_ROW in [
        dict(zip(FEATURES, [38, 0, 2, 118, 190, 0, 0, 172, 0, 0.0, 1, 0])),
        dict(zip(FEATURES, [63, 1, 4, 160, 300, 1, 2, 118, 1, 3.2, 2, 3])),
    ]])
    st.download_button("📄 Download CSV template", template.to_csv(index=False),
                       "cardioguard_template.csv", "text/csv")
    up_file = st.file_uploader("Upload CSV (must contain the 12 feature columns)", type="csv")
    if up_file is not None:
        try:
            bdf = pd.read_csv(up_file)
            missing = [c for c in FEATURES if c not in bdf.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                clean = bdf.dropna(subset=FEATURES).copy()
                if len(clean) < len(bdf):
                    st.warning(f"{len(bdf) - len(clean)} row(s) with missing values were skipped.")
                clean["Risk %"] = (predict_proba(clean) * 100).round(1)
                clean["Prediction"] = np.where(clean["Risk %"] >= 50, "Presence", "Absence")
                clean["Risk level"] = pd.cut(clean["Risk %"], [-1, 30, 60, 101],
                                             labels=["Low", "Moderate", "High"])
                m1, m2, m3, m4 = st.columns(4)
                with m1: kpi("Patients", f"{len(clean):,}")
                with m2: kpi("Avg risk", f"{clean['Risk %'].mean():.1f}%")
                with m3: kpi("High risk", f"{(clean['Risk level'] == 'High').sum():,}")
                with m4: kpi("Predicted presence", f"{(clean['Prediction'] == 'Presence').mean()*100:.0f}%")
                st.write("")
                h = go.Figure(go.Histogram(x=clean["Risk %"], nbinsx=25,
                                           marker=dict(color="#7c5cff", line=dict(color="#00d9ff", width=1))))
                h.update_layout(height=320, xaxis_title="Risk %", yaxis_title="Patients", **PLOT_BG)
                st.plotly_chart(h, width="stretch", config={"displayModeBar": False})
                st.dataframe(clean.head(1000), width="stretch", height=360)
                st.download_button("⬇️ Download predictions", clean.to_csv(index=False),
                                   "cardioguard_predictions.csv", "text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read file: {e}")

# ══════════════════ TAB 5 — MODEL INSIGHTS ══════════════════
with tab_insight:
    st.markdown("<div class='section-title'>Logistic Regression — what it learned</div>",
                unsafe_allow_html=True)
    coef = clf.coef_[0] if POS_IDX == 1 else -clf.coef_[0]
    cdf = pd.DataFrame({"Feature": FEATURES, "Coefficient": coef,
                        "Odds ratio (per 1 SD)": np.exp(coef)}).sort_values("Coefficient")
    fig = go.Figure(go.Bar(
        x=cdf["Coefficient"], y=cdf["Feature"], orientation="h",
        marker=dict(color=cdf["Coefficient"],
                    colorscale=[[0, "#00e5a0"], [.5, "#2a3550"], [1, "#ff4d7a"]],
                    cmid=0)))
    fig.update_layout(height=430, xaxis_title="Standardised coefficient", **PLOT_BG)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with st.expander("Show coefficient table"):
        st.dataframe(cdf.sort_values("Coefficient", ascending=False).round(4),
                     width="stretch", hide_index=True)

    info = load_insights()
    if info is None:
        st.info("📁 Place **train.csv** next to `app.py` to unlock hold-out metrics, ROC curve, "
                "confusion matrix and data analytics.")
    else:
        Xte, yte = info["X_test"], info["y_test"]
        ypred = model.predict(Xte)
        yprob = predict_proba(Xte)
        ybin = (yte == "Presence").astype(int)

        st.markdown("<div class='section-title'>Hold-out performance (same split as notebook)</div>",
                    unsafe_allow_html=True)
        cols = st.columns(5)
        metrics = [
            ("Accuracy", accuracy_score(yte, ypred)),
            ("Precision", precision_score(yte, ypred, pos_label="Presence", zero_division=0)),
            ("Recall", recall_score(yte, ypred, pos_label="Presence", zero_division=0)),
            ("F1-score", f1_score(yte, ypred, pos_label="Presence", zero_division=0)),
            ("ROC-AUC", roc_auc_score(ybin, yprob)),
        ]
        for c, (n, v) in zip(cols, metrics):
            with c: kpi(n, f"{v*100:.1f}%" if n != "ROC-AUC" else f"{v:.3f}", f"n = {len(yte):,}" if n == "Accuracy" else "")
        st.write("")

        a, b = st.columns(2, gap="large")
        with a:
            cm = confusion_matrix(yte, ypred, labels=["Absence", "Presence"])
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Pred Absence", "Pred Presence"], y=["Actual Absence", "Actual Presence"],
                text=cm, texttemplate="%{text:,}", colorscale="Tealgrn", showscale=False))
            fig.update_layout(title="Confusion matrix", height=380, **{**PLOT_BG, "margin": dict(l=10, r=10, t=50, b=10)})
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with b:
            fpr, tpr, _ = roc_curve(ybin, yprob)
            idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr[idx], y=tpr[idx], mode="lines", name="ROC",
                                     line=dict(color="#00d9ff", width=4),
                                     fill="tozeroy", fillcolor="rgba(0,217,255,.1)"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                     line=dict(color="#5d7189", dash="dash")))
            fig.update_layout(title=f"ROC curve (AUC = {metrics[-1][1]:.3f})", height=380,
                              xaxis_title="False positive rate", yaxis_title="True positive rate",
                              **{**PLOT_BG, "margin": dict(l=10, r=10, t=50, b=10)})
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        c, d = st.columns(2, gap="large")
        with c:
            cc = info["class_counts"]
            fig = go.Figure(go.Pie(labels=cc.index, values=cc.values, hole=.62,
                                   marker=dict(colors=["#00e5a0", "#ff4d7a"]),
                                   textinfo="label+percent"))
            fig.update_layout(title=f"Class balance ({info['n_rows']:,} rows)", height=340,
                              showlegend=False, **{**PLOT_BG, "margin": dict(l=10, r=10, t=50, b=10)})
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with d:
            ad = info["age_by_class"]
            fig = go.Figure()
            for cls, col in [("Absence", "#00e5a0"), ("Presence", "#ff4d7a")]:
                fig.add_trace(go.Violin(y=ad[ad[TARGET] == cls]["Age"], name=cls,
                                        line_color=col, meanline_visible=True, opacity=.75))
            fig.update_layout(title="Age distribution by class", height=340,
                              **{**PLOT_BG, "margin": dict(l=10, r=10, t=50, b=10)})
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        corr = info["corr"]
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                   colorscale="RdBu_r", zmid=0, text=corr.round(2).values,
                                   texttemplate="%{text}", textfont=dict(size=9)))
        fig.update_layout(title="Correlation heatmap", height=620, **{**PLOT_BG, "margin": dict(l=10, r=10, t=50, b=10)})
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ══════════════════ TAB 6 — ABOUT ══════════════════
with tab_about:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("""
<div class='glass'>
<div class='kpi-label'>Project</div>
<h3 style='margin:6px 0'>CardioGuard AI</h3>
<p style='color:#9fb3c8'>A machine-learning healthcare project that estimates the likelihood of heart
disease from 12 clinical attributes using <b>Logistic Regression</b> with standard scaling.</p>
<div class='kpi-label' style='margin-top:14px'>Pipeline</div>
<p>Raw inputs → <code>StandardScaler</code> → <code>LogisticRegression</code> → P(Presence)</p>
<div class='kpi-label' style='margin-top:14px'>Risk bands</div>
<span class='chip chip-ok'>&lt; 30% Low</span>
<span class='chip chip-warn'>30–60% Moderate</span>
<span class='chip chip-bad'>&gt; 60% High</span>
</div>""", unsafe_allow_html=True)
    with b:
        st.markdown("""
<div class='glass'>
<div class='kpi-label'>Feature guide</div>
<ul style='color:#c8d4e2;line-height:1.8;margin-top:8px'>
<li><b>Chest pain type</b> — 1 typical · 2 atypical · 3 non-anginal · 4 asymptomatic</li>
<li><b>Resting EKG</b> — 0 normal · 1 ST-T abnormality · 2 LV hypertrophy</li>
<li><b>ST depression</b> — exercise-induced ST drop vs rest</li>
<li><b>Slope of ST</b> — 1 up · 2 flat · 3 down</li>
<li><b>Vessels (fluoroscopy)</b> — number of major vessels, 0–3</li>
</ul>
</div>""", unsafe_allow_html=True)

st.markdown(
    "<div class='disclaimer'>⚠️ <b>Medical disclaimer:</b> CardioGuard AI is an educational "
    "machine-learning project. Its output is a statistical estimate, <b>not</b> a medical diagnosis. "
    "Always consult a qualified healthcare professional.</div>",
    unsafe_allow_html=True,
)
