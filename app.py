"""
CardioGuard AI — Intelligent Heart Disease Prediction System
============================================================
Streamlit app for the Logistic Regression pipeline from CardioGuard_ML.ipynb
(StandardScaler -> LogisticRegression).

RUN
    pip install -r requirements.txt
    streamlit run app.py

FILES (keep in the same folder as app.py)
    Heart_disease.pki   trained model saved with joblib   (REQUIRED; any *.pki name works)
    train.csv           original dataset                  (optional - unlocks
                                                           metrics / ROC / heatmap)

Works with Streamlit 1.30 and newer.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# 0. PAGE CONFIG + VERSION COMPATIBILITY
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _ver(v: str):
    out = []
    for part in v.split(".")[:2]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits or 0))
    return tuple(out)


# Newer Streamlit uses width="stretch"; older uses use_container_width=True.
STRETCH = (
    {"width": "stretch"} if _ver(st.__version__) >= (1, 50) else {"use_container_width": True}
)
PCFG = {"displayModeBar": False}


def show(fig):
    """Render a plotly figure on any Streamlit version."""
    st.plotly_chart(fig, config=PCFG, **STRETCH)


# ──────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = ["Heart_disease.pki", "Heart_disease__2_.pki", "Heart_disease__1_.pki",
                    "Heart_disease.pkl", "Heart_disease.joblib"]
DATA_FILE = "train.csv"

FEATURES = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
    "EKG results", "Max HR", "Exercise angina", "ST depression",
    "Slope of ST", "Number of vessels fluro",
]
TARGET = "Heart Disease"

# (min, max, dataset mean) from train.csv — used for the radar chart
RANGES = {
    "Age": (29, 77, 54.1), "BP": (94, 200, 130.5), "Cholesterol": (126, 564, 245.0),
    "Max HR": (71, 202, 152.8), "ST depression": (0.0, 6.2, 0.72),
    "Number of vessels fluro": (0, 3, 0.45),
}

CP = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
EKG = {0: "Normal", 1: "ST-T wave abnormality", 2: "LV hypertrophy"}
SLOPE = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
SEX = {0: "Female", 1: "Male"}
YN = {0: "No", 1: "Yes"}

PRESETS = {
    "low": dict(age=38, sex=0, cp=2, bp=118, chol=190, fbs=0, ekg=0, hr=172,
                angina=0, st=0.0, slope=1, vessels=0),
    "mid": dict(age=57, sex=1, cp=3, bp=134, chol=245, fbs=0, ekg=1, hr=146,
                angina=0, st=1.0, slope=2, vessels=0),
    "high": dict(age=63, sex=1, cp=4, bp=160, chol=300, fbs=1, ekg=2, hr=118,
                 angina=1, st=3.2, slope=2, vessels=3),
}

C_LOW, C_MOD, C_HIGH = "#2ee6a6", "#ffc233", "#ff5c85"
C_CYAN, C_VIOLET = "#22d3ee", "#8b7bff"

# ──────────────────────────────────────────────────────────────────────────
# 2. STYLE
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"] { font-family:'Inter',sans-serif; }

.stApp{
  background:
   radial-gradient(1100px 560px at 5% -8%, rgba(34,211,238,.15), transparent 60%),
   radial-gradient(900px 520px at 100% 0%, rgba(139,123,255,.16), transparent 55%),
   #060a13;
}
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.2rem; padding-bottom:3rem; max-width:1320px;}

/* HERO */
.hero{padding:26px 30px;border-radius:22px;margin-bottom:16px;
  background:linear-gradient(135deg,rgba(34,211,238,.10),rgba(139,123,255,.12));
  border:1px solid rgba(34,211,238,.25); box-shadow:0 12px 50px rgba(34,211,238,.07);}
.hero h1{margin:0;font-size:2.3rem;font-weight:800;letter-spacing:-.5px;
  background:linear-gradient(90deg,#22d3ee,#8b7bff 55%,#ff5c85);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero p{margin:6px 0 12px;color:#9db2c8;font-size:1rem;}
.beat{display:inline-block;animation:beat 1.3s infinite;}
@keyframes beat{0%,100%{transform:scale(1)}15%{transform:scale(1.18)}30%{transform:scale(1)}45%{transform:scale(1.1)}}
.tag{display:inline-block;margin:2px 6px 2px 0;padding:4px 12px;border-radius:999px;font-size:.76rem;
  font-weight:600;color:#bfe9f3;background:rgba(34,211,238,.09);border:1px solid rgba(34,211,238,.28);}

/* CARDS */
.card{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.10);
  border-radius:16px;padding:16px 18px;height:100%;transition:all .2s ease;}
.card:hover{border-color:rgba(34,211,238,.4);transform:translateY(-2px);}
.kl{color:#8ea4bb;font-size:.72rem;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;}
.kv{color:#fff;font-size:1.7rem;font-weight:800;line-height:1.25;margin-top:3px;}
.ks{color:#7f95ab;font-size:.8rem;margin-top:1px;}
.h2{font-size:1.15rem;font-weight:700;color:#eaf3fc;margin:10px 0 4px;}
.sub{color:#8ea4bb;font-size:.88rem;margin-bottom:10px;}

/* VERDICT */
.verdict{border-radius:18px;padding:20px 24px;margin-bottom:12px;border:1px solid;}
.verdict h2{margin:0;font-size:1.55rem;font-weight:800;}
.verdict p{margin:6px 0 0;color:#cfdae7;}
.v-low{background:linear-gradient(120deg,rgba(46,230,166,.17),rgba(46,230,166,.04));border-color:rgba(46,230,166,.5);}
.v-mod{background:linear-gradient(120deg,rgba(255,194,51,.17),rgba(255,194,51,.04));border-color:rgba(255,194,51,.5);}
.v-high{background:linear-gradient(120deg,rgba(255,92,133,.19),rgba(255,92,133,.04));border-color:rgba(255,92,133,.55);}

/* CHIPS */
.chip{display:inline-block;padding:5px 12px;border-radius:999px;font-size:.8rem;font-weight:600;margin:3px 6px 3px 0;border:1px solid;}
.ok{color:#2ee6a6;border-color:rgba(46,230,166,.45);background:rgba(46,230,166,.09);}
.warn{color:#ffc233;border-color:rgba(255,194,51,.45);background:rgba(255,194,51,.09);}
.bad{color:#ff5c85;border-color:rgba(255,92,133,.5);background:rgba(255,92,133,.10);}

/* INSIGHT LIST */
.ins{padding:10px 14px;border-radius:12px;margin:6px 0;background:rgba(255,255,255,.04);
  border-left:4px solid;color:#dce7f3;font-size:.92rem;}
.ins.up{border-color:#ff5c85;} .ins.down{border-color:#2ee6a6;} .ins.info{border-color:#22d3ee;}

/* STEPS */
.step{display:flex;gap:12px;align-items:flex-start;margin:8px 0;}
.num{min-width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,#22d3ee,#8b7bff);color:#06101c;font-weight:800;font-size:.85rem;}

.disc{margin-top:22px;padding:13px 18px;border-radius:14px;font-size:.84rem;color:#d9c98f;
  background:rgba(255,194,51,.06);border:1px solid rgba(255,194,51,.25);}

/* SIDEBAR */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#09101f,#0b1326);
  border-right:1px solid rgba(34,211,238,.14);}
section[data-testid="stSidebar"] h3{color:#22d3ee;font-size:.92rem;letter-spacing:.4px;margin-top:.6rem;}

/* TABS */
.stTabs [data-baseweb="tab-list"]{gap:6px;background:rgba(255,255,255,.035);padding:6px;border-radius:14px;flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{border-radius:10px;padding:8px 16px;font-weight:600;color:#8ea4bb;height:auto;}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,rgba(34,211,238,.2),rgba(139,123,255,.2));color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}

/* BUTTONS */
.stButton>button,.stDownloadButton>button{border-radius:12px;font-weight:600;
  border:1px solid rgba(34,211,238,.4);background:linear-gradient(90deg,rgba(34,211,238,.16),rgba(139,123,255,.16));
  color:#eaf6ff;transition:all .2s;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#22d3ee;box-shadow:0 0 18px rgba(34,211,238,.3);}
</style>
""",
    unsafe_allow_html=True,
)

PLOT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#d3e2f2"),
)


def layout(fig, h=380, **kw):
    kw.setdefault("margin", dict(l=10, r=10, t=44, b=10))
    fig.update_layout(height=h, **PLOT, **kw)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# 3. MODEL + DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Loads the first model file found next to app.py.
    Any name works: Heart_disease.pki, Heart_disease__2_.pki, model.pkl, ..."""
    for folder in dict.fromkeys([BASE_DIR, Path.cwd()]):
        found = [folder / n for n in MODEL_CANDIDATES if (folder / n).exists()]
        extra = sorted(p for ext in ("*.pki", "*.pkl", "*.joblib") for p in folder.glob(ext)
                       if p not in found)
        for p in found + extra:
            try:
                return joblib.load(p), p.name, None
            except Exception as e:  # noqa: BLE001
                last_err = (p.name, str(e))
        if found or extra:
            return None, last_err[0], last_err[1]
    return None, None, None


model, model_file, model_err = load_model()

if model is None:
    st.markdown("<div class='hero'><h1>🫀 CardioGuard AI</h1></div>", unsafe_allow_html=True)
    if model_err:
        import sklearn
        st.error(f"Model file `{model_file}` found but could not be loaded.\n\n`{model_err}`")
        st.info(f"Installed scikit-learn: **{sklearn.__version__}**. The model was saved with "
                f"**1.6.1**. Fix:  `pip install scikit-learn==1.6.1`")
    else:
        st.error("No model file (*.pki / *.pkl / *.joblib) found next to `app.py`.")
        st.info("Download it from your notebook (`joblib.dump(model, 'Heart_disease.pki')`) "
                "and place it in the same folder as app.py, then refresh.")
    st.stop()

scaler = model.named_steps["scaler"]
clf = model.named_steps["classifier"]
POS = list(model.classes_).index("Presence")
COEF = clf.coef_[0] if POS == 1 else -clf.coef_[0]
BASELINE = float(1 / (1 + np.exp(-(clf.intercept_[0] if POS == 1 else -clf.intercept_[0]))))


@st.cache_data(show_spinner="Preparing evaluation data (first time only)…")
def load_insights():
    from sklearn.model_selection import train_test_split

    path = BASE_DIR / DATA_FILE
    if not path.exists():
        path = Path.cwd() / DATA_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path).drop_duplicates()
    X, y = df[FEATURES], df[TARGET]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    corr = df[FEATURES + ["Thallium"]].copy()
    corr["Target"] = (df[TARGET] == "Presence").astype(int)
    return dict(
        X_test=X_te, y_test=y_te, n_rows=len(df),
        corr=corr.sample(min(len(corr), 60000), random_state=1).corr(),
        counts=df[TARGET].value_counts(),
        age=df[["Age", TARGET]].sample(min(len(df), 40000), random_state=1),
    )


@st.cache_data(show_spinner="Evaluating model on hold-out data…")
def evaluate():
    """Heavy metrics computed ONCE and cached, so sliders stay instant."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score, roc_curve)
    info = load_insights()
    if info is None:
        return None
    yte, Xte = info["y_test"], info["X_test"]
    ypred = model.predict(Xte)
    yprob = model.predict_proba(Xte[FEATURES])[:, POS]
    ybin = (yte == "Presence").astype(int)
    fpr, tpr, _ = roc_curve(ybin, yprob)
    idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int)
    return dict(
        n=len(yte),
        acc=accuracy_score(yte, ypred),
        prec=precision_score(yte, ypred, pos_label="Presence"),
        rec=recall_score(yte, ypred, pos_label="Presence"),
        f1=f1_score(yte, ypred, pos_label="Presence"),
        auc=roc_auc_score(ybin, yprob),
        cm=confusion_matrix(yte, ypred, labels=["Absence", "Presence"]),
        fpr=fpr[idx], tpr=tpr[idx],
    )


# ──────────────────────────────────────────────────────────────────────────
# 4. HELPERS
# ──────────────────────────────────────────────────────────────────────────
def proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, POS]


def band(p: float):
    if p < 0.30:
        return "Low Risk", "v-low", C_LOW, "🟢"
    if p < 0.60:
        return "Moderate Risk", "v-mod", C_MOD, "🟡"
    return "High Risk", "v-high", C_HIGH, "🔴"


def contributions(row: pd.DataFrame) -> pd.DataFrame:
    z = (row[FEATURES].values[0] - scaler.mean_) / scaler.scale_
    return pd.DataFrame({"Feature": FEATURES, "Value": row[FEATURES].values[0],
                         "Contribution": COEF * z}).sort_values("Contribution")


def fmt_val(feat: str, v) -> str:
    v = float(v)
    maps = {"Sex": SEX, "Chest pain type": CP, "EKG results": EKG, "Slope of ST": SLOPE,
            "FBS over 120": YN, "Exercise angina": YN}
    if feat in maps:
        return maps[feat].get(int(v), str(v))
    return f"{v:g}"


def gauge(p: float) -> go.Figure:
    color = band(p)[2]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=p * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 52, "color": color}},
        delta={"reference": BASELINE * 100, "valueformat": "+.1f", "suffix": " vs avg patient",
               "increasing": {"color": C_HIGH}, "decreasing": {"color": C_LOW},
               "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#5d7189"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.03)", "borderwidth": 0,
            "steps": [{"range": [0, 30], "color": "rgba(46,230,166,.16)"},
                      {"range": [30, 60], "color": "rgba(255,194,51,.16)"},
                      {"range": [60, 100], "color": "rgba(255,92,133,.18)"}],
        }))
    return layout(fig, 290, margin=dict(l=25, r=25, t=30, b=0))


def radar(row: pd.DataFrame) -> go.Figure:
    keys = list(RANGES)
    nz = lambda k, v: float(np.clip((v - RANGES[k][0]) / (RANGES[k][1] - RANGES[k][0]), 0, 1))
    pat = [nz(k, row[k].iloc[0]) for k in keys]
    avg = [nz(k, RANGES[k][2]) for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=avg + avg[:1], theta=keys + keys[:1], name="Dataset average",
                                  line=dict(color=C_VIOLET, dash="dot"),
                                  fill="toself", fillcolor="rgba(139,123,255,.12)"))
    fig.add_trace(go.Scatterpolar(r=pat + pat[:1], theta=keys + keys[:1], name="This patient",
                                  line=dict(color=C_CYAN, width=3),
                                  fill="toself", fillcolor="rgba(34,211,238,.18)"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 1], showticklabels=False,
                                                 gridcolor="rgba(255,255,255,.1)"),
                                 angularaxis=dict(gridcolor="rgba(255,255,255,.1)")),
                      legend=dict(orientation="h", y=-0.08))
    return layout(fig, 400)


def vital_flags(r: pd.Series):
    """Rule-of-thumb context only — NOT model inputs."""
    pct = r["Max HR"] / max(220 - r["Age"], 1) * 100
    lv = lambda x, a, b: "ok" if x < a else "warn" if x < b else "bad"
    return [
        ("Blood pressure", f"{r['BP']:.0f} mmHg", lv(r["BP"], 130, 140)),
        ("Cholesterol", f"{r['Cholesterol']:.0f} mg/dL", lv(r["Cholesterol"], 200, 240)),
        ("Max HR vs age-predicted", f"{pct:.0f}%", "ok" if pct >= 85 else "warn" if pct >= 70 else "bad"),
        ("ST depression", f"{r['ST depression']:.1f}", lv(r["ST depression"], 1, 2)),
        ("Fasting sugar", ">120 mg/dL" if r["FBS over 120"] else "≤120 mg/dL", "bad" if r["FBS over 120"] else "ok"),
        ("Exercise angina", "Present" if r["Exercise angina"] else "Absent", "bad" if r["Exercise angina"] else "ok"),
    ]


def report_text(r: pd.Series, p: float) -> str:
    lab = band(p)[0]
    rows = [f"{f:<26}: {fmt_val(f, r[f])}" for f in FEATURES]
    return "\n".join([
        "CARDIOGUARD AI — PATIENT RISK REPORT", "=" * 46,
        f"Generated : {datetime.now():%d %b %Y, %H:%M}",
        f"Result    : {lab}",
        f"Probability of heart disease : {p*100:.1f}%",
        f"Average-patient baseline     : {BASELINE*100:.1f}%", "",
        "INPUT PARAMETERS", "-" * 46, *rows, "",
        "DISCLAIMER", "Educational ML project (Logistic Regression). Not a medical diagnosis.",
        "Consult a qualified doctor for clinical decisions.",
    ])


def kpi(label, value, sub=""):
    st.markdown(f"<div class='card'><div class='kl'>{label}</div><div class='kv'>{value}</div>"
                f"<div class='ks'>{sub}</div></div>", unsafe_allow_html=True)


def head(title, sub=""):
    st.markdown(f"<div class='h2'>{title}</div>" + (f"<div class='sub'>{sub}</div>" if sub else ""),
                unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR — INPUTS
# ──────────────────────────────────────────────────────────────────────────
for k, v in PRESETS["mid"].items():
    st.session_state.setdefault(k, v)


def set_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:4px 0 2px'><span class='beat' style='font-size:2.4rem'>🫀</span>"
        "<div style='font-weight:800;font-size:1.25rem;color:#22d3ee'>CardioGuard AI</div>"
        "<div style='color:#7f95ab;font-size:.8rem'>Enter patient details — results update live</div></div>",
        unsafe_allow_html=True)

    st.markdown("### ⚡ Quick presets")
    b1, b2, b3 = st.columns(3)
    b1.button("🟢 Low", on_click=set_preset, args=("low",), **STRETCH)
    b2.button("🟡 Mid", on_click=set_preset, args=("mid",), **STRETCH)
    b3.button("🔴 High", on_click=set_preset, args=("high",), **STRETCH)

    st.markdown("### 👤 1 · Demographics")
    age = st.slider("Age (years)", 29, 77, key="age")
    sex = st.radio("Sex", [0, 1], format_func=SEX.get, horizontal=True, key="sex")

    st.markdown("### 🩸 2 · Vitals & labs")
    bp = st.slider("Resting blood pressure (mmHg)", 94, 200, key="bp",
                   help="Normal is below 120/80. 130+ is considered elevated.")
    chol = st.slider("Serum cholesterol (mg/dL)", 126, 564, key="chol",
                     help="Below 200 is desirable.")
    fbs = st.radio("Fasting blood sugar > 120 mg/dL", [0, 1], format_func=YN.get,
                   horizontal=True, key="fbs")

    st.markdown("### 🫀 3 · Cardiac tests")
    cp = st.selectbox("Chest pain type", list(CP), format_func=CP.get, key="cp")
    ekg = st.selectbox("Resting EKG result", list(EKG), format_func=EKG.get, key="ekg")
    hr = st.slider("Maximum heart rate (bpm)", 71, 202, key="hr",
                   help="Highest heart rate reached during the exercise test.")
    angina = st.radio("Exercise-induced angina", [0, 1], format_func=YN.get,
                      horizontal=True, key="angina")
    st_dep = st.slider("ST depression", 0.0, 6.2, step=0.1, key="st",
                       help="ST-segment drop induced by exercise relative to rest.")
    slope = st.selectbox("Slope of peak-exercise ST", list(SLOPE), format_func=SLOPE.get, key="slope")
    vessels = st.select_slider("Major vessels coloured by fluoroscopy", [0, 1, 2, 3], key="vessels")

    st.caption(f"Model: `{model_file}` · 12 features")

patient = pd.DataFrame([{
    "Age": age, "Sex": sex, "Chest pain type": cp, "BP": bp, "Cholesterol": chol,
    "FBS over 120": fbs, "EKG results": ekg, "Max HR": hr, "Exercise angina": angina,
    "ST depression": st_dep, "Slope of ST": slope, "Number of vessels fluro": vessels,
}])
prob = float(proba(patient)[0])
label, vclass, vcolor, emoji = band(prob)
contrib = contributions(patient)

# ──────────────────────────────────────────────────────────────────────────
# 6. HERO
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='hero'><h1><span class='beat'>🫀</span> CardioGuard AI</h1>"
    "<p>Intelligent heart disease risk prediction — predict early, protect every beat.</p>"
    "<span class='tag'>Logistic Regression</span><span class='tag'>12 clinical features</span>"
    "<span class='tag'>Explainable</span><span class='tag'>Live predictions</span></div>",
    unsafe_allow_html=True)

t_risk, t_why, t_what, t_batch, t_model, t_help = st.tabs([
    "🩺 Risk Assessment", "🔬 Why this result?", "🧪 What-If Lab",
    "📂 Batch Prediction", "📊 Model Insights", "❓ Guide"])

# ═════════════ TAB 1 — RISK ═════════════
with t_risk:
    c1, c2 = st.columns([1.05, 1], gap="large")
    with c1:
        show(gauge(prob))
        st.markdown(
            "<div style='text-align:center'><span class='chip ok'>0–30% Low</span>"
            "<span class='chip warn'>30–60% Moderate</span><span class='chip bad'>60–100% High</span></div>",
            unsafe_allow_html=True)
    with c2:
        msg = {"v-low": "Inputs look favourable. Keep up healthy habits and routine check-ups.",
               "v-mod": "Some indicators deserve attention. A consultation and lifestyle review is sensible.",
               "v-high": "Several indicators are associated with heart disease. A clinical evaluation is advised."}[vclass]
        st.markdown(f"<div class='verdict {vclass}'><h2>{emoji} {label}</h2><p>{msg}</p></div>",
                    unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1: kpi("Presence", f"{prob*100:.1f}%", "heart disease")
        with k2: kpi("Absence", f"{(1-prob)*100:.1f}%", "no disease")
        with k3: kpi("Avg patient", f"{BASELINE*100:.1f}%", "baseline risk")
        st.write("")
        st.download_button("⬇️ Download patient report (.txt)", report_text(patient.iloc[0], prob),
                           f"cardioguard_report_{datetime.now():%Y%m%d_%H%M}.txt", "text/plain", **STRETCH)

    head("🧭 Key findings", "Auto-generated from this patient's inputs.")
    up = contrib[contrib["Contribution"] > 0.05].sort_values("Contribution", ascending=False).head(3)
    dn = contrib[contrib["Contribution"] < -0.05].head(3)
    f1, f2 = st.columns(2, gap="large")
    with f1:
        st.markdown("**Raising risk**")
        if up.empty:
            st.markdown("<div class='ins info'>No strong risk-raising factor.</div>", unsafe_allow_html=True)
        for _, r in up.iterrows():
            st.markdown(f"<div class='ins up'>🔺 <b>{r['Feature']}</b> = {fmt_val(r['Feature'], r['Value'])}"
                        f" &nbsp;<span style='color:#ff5c85'>({r['Contribution']:+.2f})</span></div>",
                        unsafe_allow_html=True)
    with f2:
        st.markdown("**Lowering risk**")
        if dn.empty:
            st.markdown("<div class='ins info'>No strong protective factor.</div>", unsafe_allow_html=True)
        for _, r in dn.iterrows():
            st.markdown(f"<div class='ins down'>🔻 <b>{r['Feature']}</b> = {fmt_val(r['Feature'], r['Value'])}"
                        f" &nbsp;<span style='color:#2ee6a6'>({r['Contribution']:+.2f})</span></div>",
                        unsafe_allow_html=True)

    head("🩻 Clinical snapshot", "Rule-of-thumb thresholds for context only — they are not model inputs.")
    st.markdown("".join(f"<span class='chip {lvl}'>{n}: {v}</span>"
                        for n, v, lvl in vital_flags(patient.iloc[0])), unsafe_allow_html=True)

    head("🕸️ Patient vs dataset average")
    show(radar(patient))

# ═════════════ TAB 2 — EXPLAIN ═════════════
with t_why:
    head("Why did the model give this result?",
         "Each bar shows how much a feature pushes the log-odds of heart disease up (red) or down (green) "
         "compared with an average patient.")
    cols = [C_HIGH if v > 0 else C_LOW for v in contrib["Contribution"]]
    fig = go.Figure(go.Bar(
        x=contrib["Contribution"], y=contrib["Feature"], orientation="h",
        marker=dict(color=cols),
        text=[f"{v:+.2f}" for v in contrib["Contribution"]], textposition="outside",
        customdata=[fmt_val(f, v) for f, v in zip(contrib["Feature"], contrib["Value"])],
        hovertemplate="<b>%{y}</b><br>Value: %{customdata}<br>Effect: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color="rgba(255,255,255,.35)")
    layout(fig, 480, xaxis_title="← lowers risk          raises risk →")
    show(fig)
    total = contrib["Contribution"].sum()
    st.markdown(
        f"<div class='ins info'>Sum of all effects: <b>{total:+.2f}</b> log-odds → predicted risk "
        f"<b>{prob*100:.1f}%</b> (average patient: {BASELINE*100:.1f}%).</div>", unsafe_allow_html=True)

# ═════════════ TAB 3 — WHAT-IF ═════════════
with t_what:
    head("What-If Lab", "Pick one parameter and see how the risk changes while everything else stays fixed.")
    sweeps = {"Age": np.arange(29, 78, 1), "BP": np.arange(94, 201, 2),
              "Cholesterol": np.arange(126, 565, 6), "Max HR": np.arange(71, 203, 2),
              "ST depression": np.round(np.arange(0, 6.3, 0.1), 1)}
    feat = st.selectbox("Parameter to vary", list(sweeps))
    grid = sweeps[feat]
    sim = pd.concat([patient] * len(grid), ignore_index=True)
    sim[feat] = grid
    pr = proba(sim) * 100
    cur = float(patient[feat].iloc[0])

    fig = go.Figure()
    for y0, y1, c in [(0, 30, "46,230,166"), (30, 60, "255,194,51"), (60, 100, "255,92,133")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=f"rgba({c},.07)", line_width=0)
    fig.add_trace(go.Scatter(x=grid, y=pr, mode="lines", name="Predicted risk",
                             line=dict(color=C_CYAN, width=4, shape="spline"),
                             fill="tozeroy", fillcolor="rgba(34,211,238,.08)"))
    fig.add_trace(go.Scatter(x=[cur], y=[prob * 100], mode="markers", name="Current patient",
                             marker=dict(size=15, color=vcolor, line=dict(color="#fff", width=2))))
    layout(fig, 430, xaxis_title=feat, yaxis_title="Heart-disease probability (%)", yaxis=dict(range=[0, 100]))
    show(fig)
    st.info(f"Across the full {feat} range, predicted risk moves from **{pr[0]:.1f}%** to "
            f"**{pr[-1]:.1f}%** ({pr[-1]-pr[0]:+.1f} points).")

# ═════════════ TAB 4 — BATCH ═════════════
with t_batch:
    head("Batch prediction", "Upload a CSV with the 12 feature columns to score many patients at once.")
    tpl = pd.DataFrame([dict(zip(FEATURES, [38, 0, 2, 118, 190, 0, 0, 172, 0, 0.0, 1, 0])),
                        dict(zip(FEATURES, [63, 1, 4, 160, 300, 1, 2, 118, 1, 3.2, 2, 3]))])
    st.download_button("📄 Download CSV template", tpl.to_csv(index=False),
                       "cardioguard_template.csv", "text/csv")
    f = st.file_uploader("Upload CSV", type="csv")
    if f is not None:
        try:
            bdf = pd.read_csv(f)
            miss = [c for c in FEATURES if c not in bdf.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                ok = bdf.dropna(subset=FEATURES).copy()
                if len(ok) < len(bdf):
                    st.warning(f"{len(bdf)-len(ok)} row(s) with missing values were skipped.")
                num = ok[FEATURES].apply(pd.to_numeric, errors="coerce")
                if num.isna().any().any():
                    st.error("Some feature values are not numeric. Please clean the CSV and re-upload.")
                    st.stop()
                ok[FEATURES] = num
                lim = {"Age": (29, 77), "BP": (94, 200), "Cholesterol": (126, 564),
                       "Max HR": (71, 202), "ST depression": (0, 6.2)}
                oor = sum(int(((ok[c] < lo) | (ok[c] > hi)).sum()) for c, (lo, hi) in lim.items())
                if oor:
                    st.warning(f"{oor} value(s) lie outside the range the model was trained on "
                               "(e.g. Age 29–77, BP 94–200). Predictions for them are less reliable.")
                ok["Risk %"] = (proba(ok) * 100).round(1)
                ok["Prediction"] = np.where(ok["Risk %"] >= 50, "Presence", "Absence")
                ok["Risk level"] = pd.cut(ok["Risk %"], [-1, 30, 60, 101],
                                          labels=["Low", "Moderate", "High"]).astype(str)
                m = st.columns(4)
                with m[0]: kpi("Patients", f"{len(ok):,}")
                with m[1]: kpi("Average risk", f"{ok['Risk %'].mean():.1f}%")
                with m[2]: kpi("High risk", f"{(ok['Risk level']=='High').sum():,}")
                with m[3]: kpi("Predicted presence", f"{(ok['Prediction']=='Presence').mean()*100:.0f}%")
                st.write("")
                h = go.Figure(go.Histogram(x=ok["Risk %"], nbinsx=25,
                                           marker=dict(color=C_VIOLET, line=dict(color=C_CYAN, width=1))))
                show(layout(h, 320, xaxis_title="Risk %", yaxis_title="Patients"))
                st.dataframe(ok.head(1000), height=360, **STRETCH)
                st.download_button("⬇️ Download predictions", ok.to_csv(index=False),
                                   "cardioguard_predictions.csv", "text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not process file: {e}")

# ═════════════ TAB 5 — MODEL INSIGHTS ═════════════
with t_model:
    head("What the model learned", "Standardised logistic-regression coefficients (red raises risk, green lowers it).")
    cdf = pd.DataFrame({"Feature": FEATURES, "Coefficient": COEF,
                        "Odds ratio (per 1 SD)": np.exp(COEF)}).sort_values("Coefficient")
    fig = go.Figure(go.Bar(x=cdf["Coefficient"], y=cdf["Feature"], orientation="h",
                           marker=dict(color=[C_HIGH if v > 0 else C_LOW for v in cdf["Coefficient"]])))
    show(layout(fig, 430, xaxis_title="Standardised coefficient"))
    with st.expander("Coefficient table"):
        st.dataframe(cdf.sort_values("Coefficient", ascending=False).round(4), **STRETCH)

    info = load_insights()
    if info is None:
        st.info("📁 Put **train.csv** next to `app.py` to unlock hold-out metrics, ROC curve, "
                "confusion matrix and data analytics.")
    else:
        ev = evaluate()
        auc = ev["auc"]
        head("Hold-out performance", f"Same 75/25 split as the notebook · {ev['n']:,} unseen rows.")
        vals = [("Accuracy", f"{ev['acc']*100:.1f}%"), ("Precision", f"{ev['prec']*100:.1f}%"),
                ("Recall", f"{ev['rec']*100:.1f}%"), ("F1-score", f"{ev['f1']*100:.1f}%"),
                ("ROC-AUC", f"{auc:.3f}")]
        for col, (n, v) in zip(st.columns(5), vals):
            with col: kpi(n, v)
        st.write("")

        a, b = st.columns(2, gap="large")
        with a:
            cm = ev["cm"]
            fig = go.Figure(go.Heatmap(z=cm, x=["Pred Absence", "Pred Presence"],
                                       y=["Actual Absence", "Actual Presence"], text=cm,
                                       texttemplate="%{text:,}", colorscale="Tealgrn", showscale=False))
            fig.update_yaxes(autorange="reversed")
            show(layout(fig, 360, title="Confusion matrix"))
        with b:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ev["fpr"], y=ev["tpr"], mode="lines", name="ROC",
                                     line=dict(color=C_CYAN, width=4),
                                     fill="tozeroy", fillcolor="rgba(34,211,238,.1)"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                     line=dict(color="#5d7189", dash="dash")))
            show(layout(fig, 360, title=f"ROC curve (AUC {auc:.3f})",
                        xaxis_title="False positive rate", yaxis_title="True positive rate"))

        c, d = st.columns(2, gap="large")
        with c:
            cc = info["counts"]
            fig = go.Figure(go.Pie(labels=cc.index, values=cc.values, hole=.62,
                                   marker=dict(colors=[C_LOW, C_HIGH]), textinfo="label+percent"))
            show(layout(fig, 330, title=f"Class balance ({info['n_rows']:,} rows)", showlegend=False))
        with d:
            ad = info["age"]
            fig = go.Figure()
            for cls, col in [("Absence", C_LOW), ("Presence", C_HIGH)]:
                fig.add_trace(go.Violin(y=ad[ad[TARGET] == cls]["Age"], name=cls, line_color=col,
                                        meanline_visible=True, opacity=.75))
            show(layout(fig, 330, title="Age distribution by class"))

        corr = info["corr"]
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu_r",
                                   zmid=0, text=corr.round(2).values, texttemplate="%{text}",
                                   textfont=dict(size=9)))
        fig.update_yaxes(autorange="reversed")
        show(layout(fig, 620, title="Correlation heatmap"))

# ═════════════ TAB 6 — GUIDE ═════════════
with t_help:
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("""
<div class='card'><div class='kl'>How to use</div>
<div class='step'><div class='num'>1</div><div>Enter patient details in the <b>sidebar</b> (or tap a preset).</div></div>
<div class='step'><div class='num'>2</div><div>Read the <b>Risk Assessment</b> — gauge, level and key findings update instantly.</div></div>
<div class='step'><div class='num'>3</div><div>Open <b>Why this result?</b> to see which inputs drove the prediction.</div></div>
<div class='step'><div class='num'>4</div><div>Use the <b>What-If Lab</b> to test how changing one value affects risk.</div></div>
<div class='step'><div class='num'>5</div><div>Use <b>Batch Prediction</b> for CSV files, and <b>Download report</b> to save results.</div></div>
</div>""", unsafe_allow_html=True)
    with g2:
        st.markdown("""
<div class='card'><div class='kl'>Feature guide</div>
<ul style='color:#cfdae7;line-height:1.85;margin-top:8px'>
<li><b>Chest pain</b> — 1 typical · 2 atypical · 3 non-anginal · 4 asymptomatic</li>
<li><b>Resting EKG</b> — 0 normal · 1 ST-T abnormality · 2 LV hypertrophy</li>
<li><b>ST depression</b> — exercise-induced ST drop versus rest</li>
<li><b>Slope of ST</b> — 1 up · 2 flat · 3 down</li>
<li><b>Vessels</b> — major vessels (0–3) seen on fluoroscopy</li>
</ul>
<div class='kl' style='margin-top:10px'>Pipeline</div>
<p style='color:#cfdae7'>Inputs → <code>StandardScaler</code> → <code>LogisticRegression</code> → P(Presence)</p>
</div>""", unsafe_allow_html=True)

st.markdown(
    "<div class='disc'>⚠️ <b>Medical disclaimer:</b> CardioGuard AI is an educational machine-learning "
    "project. Its output is a statistical estimate, <b>not</b> a medical diagnosis. Always consult a "
    "qualified healthcare professional.</div>", unsafe_allow_html=True)
