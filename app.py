"""
CardioGuard AI — Intelligent Heart Disease Prediction System
============================================================
Streamlit app for the Logistic Regression pipeline from CardioGuard_ML.ipynb
(StandardScaler -> LogisticRegression).

RUN
    pip install -r requirements.txt
    streamlit run app.py

FILES (keep in the same folder as app.py)
    Heart_disease_v2.pki  improved model from train_model.py  (recommended)
    Heart_disease*.pki    older 12-feature model also works   (any *.pki name works)
    train.csv             original dataset  (optional - unlocks metrics / ROC / heatmap)

The app adapts automatically: if the model was trained with Thallium it shows
that input, otherwise it hides it.

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
MODEL_CANDIDATES = ["Heart_disease_v2.pki", "Heart_disease.pki", "Heart_disease__2_.pki",
                    "Heart_disease__1_.pki", "Heart_disease.pkl", "Heart_disease.joblib"]
DATA_FILE = "train.csv"

ALL_FEATURES = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
    "EKG results", "Max HR", "Exercise angina", "ST depression",
    "Slope of ST", "Number of vessels fluro", "Thallium",
]
FEATURES = list(ALL_FEATURES)   # narrowed to what the loaded model needs
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
THAL = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}

# Real disease rates in train.csv (used to explain the model to students)
RATE_CP = {1: 10.8, 2: 16.2, 3: 19.1, 4: 69.7}
RATE_THAL = {3: 19.8, 6: 68.6, 7: 81.5}

PRESETS = {
    "low": dict(age=38, sex=0, cp=2, bp=118, chol=190, fbs=0, ekg=0, hr=172,
                angina=0, st=0.0, slope=1, vessels=0, thal=3),
    "mid": dict(age=58, sex=1, cp=3, bp=136, chol=250, fbs=0, ekg=2, hr=140,
                angina=0, st=1.5, slope=2, vessels=0, thal=3),
    "high": dict(age=63, sex=1, cp=4, bp=160, chol=300, fbs=1, ekg=2, hr=118,
                 angina=1, st=3.2, slope=2, vessels=3, thal=7),
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
    """Loads the first model file that works. Returns (model, name, error, skipped_files).
    Any name works: Heart_disease_v2.pki, Heart_disease__2_.pki, model.pkl, ..."""
    skipped, last = [], (None, None)
    for folder in dict.fromkeys([BASE_DIR, Path.cwd()]):
        found = [folder / n for n in MODEL_CANDIDATES if (folder / n).exists()]
        extra = sorted(p for ext in ("*.pki", "*.pkl", "*.joblib") for p in folder.glob(ext)
                       if p not in found)
        for p in found + extra:
            try:
                m = joblib.load(p)
                if not hasattr(m, "predict_proba"):
                    raise ValueError("file is not a classifier")
                return m, p.name, None, skipped
            except Exception as e:  # noqa: BLE001
                skipped.append((p.name, str(e)[:160]))
                last = (p.name, str(e))
        if skipped:
            return None, last[0], last[1], skipped
    return None, None, None, skipped


model, model_file, model_err, skipped_models = load_model()

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
SIGN = 1.0 if POS == 1 else -1.0
COEF = SIGN * clf.coef_[0]
INTERCEPT = SIGN * float(clf.intercept_[0])
BASELINE = float(1 / (1 + np.exp(-INTERCEPT)))
RAW_NAMES = list(getattr(model, "feature_names_in_", ALL_FEATURES[:12]))   # columns the model sees
FEATURES = [f for f in ALL_FEATURES
            if any(n == f or n.startswith(f + "_") for n in RAW_NAMES)]       # inputs the user fills
HAS_THAL = "Thallium" in FEATURES

if skipped_models:
    st.warning("⚠️ " + " · ".join(f"`{n}` could not be loaded ({e})" for n, e in skipped_models)
               + f" — using `{model_file}` instead. To fix: run `python train_model.py` in this same "
                 "environment, or install the scikit-learn version the model was saved with.")


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Raw inputs -> the exact columns the model was trained on (adds one 0/1 column per
    category label, e.g. 'Chest pain type_4'). Works for both the old and the v2 model."""
    out = {}
    for n in RAW_NAMES:
        if n in df.columns:
            out[n] = pd.to_numeric(df[n])
        else:
            f, _, v = n.rpartition("_")
            out[n] = (df[f].astype(float) == float(v)).astype(int)
    return pd.DataFrame(out, index=df.index)


def to_z(df: pd.DataFrame) -> np.ndarray:
    """Raw inputs -> the standardised numbers the logistic regression actually sees."""
    return scaler.transform(design(df))


def term_names() -> list:
    """Names of the columns the regression sees (one per weight)."""
    return list(RAW_NAMES)


def owner_of(term: str) -> str:
    """Which original input does a regression column belong to?"""
    for f in sorted(FEATURES, key=len, reverse=True):
        if term == f or term.startswith(f + "_"):
            return f
    return term


def pretty_term(term: str) -> str:
    f = owner_of(term)
    if term == f:
        return f
    val = term[len(f) + 1:]
    maps = {"Chest pain type": CP, "EKG results": EKG, "Slope of ST": SLOPE, "Thallium": THAL}
    try:
        return f"{f}: {maps[f][int(float(val))]}"
    except Exception:  # noqa: BLE001
        return f"{f} = {val}"


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
    corr = df[ALL_FEATURES].copy()
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
    ypred = model.predict(design(Xte))
    yprob = proba(Xte)
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
    return model.predict_proba(design(df))[:, POS]


def band(p: float):
    if p < 0.30:
        return "Low Risk", "v-low", C_LOW, "🟢"
    if p < 0.60:
        return "Moderate Risk", "v-mod", C_MOD, "🟡"
    return "High Risk", "v-high", C_HIGH, "🔴"


def contributions(row: pd.DataFrame) -> pd.DataFrame:
    """Exact push each original input gives to the log-odds, versus an average patient.
    (one-hot columns are added back together so every input has ONE bar)"""
    z = to_z(row)[0]
    parts = pd.DataFrame({"Feature": [owner_of(t) for t in term_names()], "Contribution": COEF * z})
    agg = parts.groupby("Feature", sort=False)["Contribution"].sum()
    out = pd.DataFrame({"Feature": FEATURES, "Value": row[FEATURES].values[0]})
    out["Contribution"] = out["Feature"].map(agg).fillna(0.0).values
    return out.sort_values("Contribution")


def pct(p: float) -> str:
    """Never claim certainty: show >99% / <1% instead of 100.0% / 0.0%."""
    return ">99%" if p > 0.99 else "<1%" if p < 0.01 else f"{p*100:.1f}%"


def edge_count(row: pd.Series) -> int:
    """How many numeric inputs sit at the extreme 10% edge of the training range."""
    n = 0
    for k, (lo, hi, _) in RANGES.items():
        if k in row.index and k != "Number of vessels fluro":
            span = hi - lo
            n += int(row[k] <= lo + 0.1 * span or row[k] >= hi - 0.1 * span)
    return n


def fmt_val(feat: str, v) -> str:
    v = float(v)
    maps = {"Sex": SEX, "Chest pain type": CP, "EKG results": EKG, "Slope of ST": SLOPE,
            "FBS over 120": YN, "Exercise angina": YN, "Thallium": THAL}
    if feat in maps:
        return maps[feat].get(int(v), str(v))
    return f"{v:g}"


def gauge(p: float) -> go.Figure:
    color = band(p)[2]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=float(np.clip(p * 100, 1, 99)),
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
        f"Probability of heart disease : {pct(p)}",
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
    age = st.slider("Age (years)", 29, 77, key="age",
                    help="Risk goes up with age because arteries slowly stiffen and narrow.")
    sex = st.radio("Sex", [0, 1], format_func=SEX.get, horizontal=True, key="sex",
                   help="A statistical pattern in this dataset (about 56% of men vs 18% of women had "
                        "disease). It is not a judgement about any person.")

    st.markdown("### 🩸 2 · Vitals & labs")
    bp = st.slider("Resting blood pressure (mmHg)", 94, 200, key="bp",
                   help="How hard blood pushes on artery walls while resting. Normal is about 120. "
                        "Higher means the heart works harder.")
    chol = st.slider("Serum cholesterol (mg/dL)", 126, 564, key="chol",
                     help="A fatty substance in blood. Too much can build up inside arteries. "
                          "Below 200 is considered good.")
    fbs = st.radio("Fasting blood sugar > 120 mg/dL", [0, 1], format_func=YN.get,
                   horizontal=True, key="fbs",
                   help="Blood sugar measured after not eating. Over 120 can point to diabetes, "
                        "which damages blood vessels.")

    st.markdown("### 🫀 3 · Cardiac tests")
    cp = st.selectbox("Chest pain type", list(CP), format_func=lambda k: f"{k} · {CP[k]}", key="cp",
                      help="Type of chest pain felt. The numbers are LABELS, not a scale. "
                           "Surprise: 4 (Asymptomatic = no classic chest pain) has the HIGHEST "
                           "disease rate in this data, because these patients were sent for heart "
                           "tests anyway. See the Guide tab for the full explanation.")
    st.caption(f"📊 In the training data, **{RATE_CP[cp]:.0f}%** of patients with this chest-pain type "
               "had heart disease.")
    if cp == 4:
        st.info("Asymptomatic ≠ safe. Many heart problems are silent, so this group has the highest "
                "disease rate in the data.")
    ekg = st.selectbox("Resting EKG result", list(EKG), format_func=lambda k: f"{k} · {EKG[k]}",
                       key="ekg", help="The heart's electrical recording while resting.")
    hr = st.slider("Maximum heart rate (bpm)", 71, 202, key="hr",
                   help="Highest pulse reached during the exercise test. A healthy heart can speed up "
                        "a lot, so HIGHER is better (lowers risk).")
    angina = st.radio("Exercise-induced angina", [0, 1], format_func=YN.get,
                      horizontal=True, key="angina",
                      help="Did the exercise test cause chest pain? 'Yes' is a warning sign.")
    st_dep = st.slider("ST depression", 0.0, 6.2, step=0.1, key="st",
                       help="How far a part of the EKG wave dips during exercise compared with rest. "
                            "A bigger dip means the heart muscle is struggling.")
    slope = st.selectbox("Slope of peak-exercise ST", list(SLOPE),
                         format_func=lambda k: f"{k} · {SLOPE[k]}", key="slope",
                         help="Direction of the ST part of the EKG at peak exercise. "
                              "Flat or downsloping is more worrying than upsloping.")
    vessels = st.select_slider("Major vessels coloured by fluoroscopy", [0, 1, 2, 3], key="vessels",
                               help="Number of main blood vessels (0-3) that show blockage on a "
                                    "special X-ray. More blocked vessels = higher risk.")
    if HAS_THAL:
        thal = st.selectbox("Thallium stress test", list(THAL),
                            format_func=lambda k: f"{k} · {THAL[k]}", key="thal",
                            help="A scan that shows blood flow to the heart. Normal = good flow. "
                                 "Fixed defect = permanent damage. Reversible defect = blood flow "
                                 "drops under stress.")
        st.caption(f"📊 In the training data, **{RATE_THAL[thal]:.0f}%** of patients with this result "
                   "had heart disease.")
    else:
        thal = 3

    st.caption(f"Model: `{model_file}` · {len(FEATURES)} features")

patient = pd.DataFrame([{
    "Age": age, "Sex": sex, "Chest pain type": cp, "BP": bp, "Cholesterol": chol,
    "FBS over 120": fbs, "EKG results": ekg, "Max HR": hr, "Exercise angina": angina,
    "ST depression": st_dep, "Slope of ST": slope, "Number of vessels fluro": vessels,
    "Thallium": thal,
}])[FEATURES]
prob = float(proba(patient)[0])
label, vclass, vcolor, emoji = band(prob)
contrib = contributions(patient)

# ──────────────────────────────────────────────────────────────────────────
# 6. HERO
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='hero'><h1><span class='beat'>🫀</span> CardioGuard AI</h1>"
    "<p>Intelligent heart disease risk prediction — predict early, protect every beat.</p>"
    "<span class='tag'>Logistic Regression</span><span class='tag'>" + str(len(FEATURES)) + " clinical features</span>"
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
        with k1: kpi("Presence", pct(prob), "heart disease")
        with k2: kpi("Absence", pct(1 - prob), "no disease")
        with k3: kpi("Avg patient", f"{BASELINE*100:.1f}%", "baseline risk")
        if prob > 0.99 or prob < 0.01 or edge_count(patient.iloc[0]) >= 3:
            if edge_count(patient.iloc[0]) >= 3:
                st.warning("🔎 **Reality check:** this is an extreme combination. Several inputs sit at "
                           "the edge of what the model saw during training, and real patients rarely "
                           "look like this. Read the result as 'very high / very low', not as an exact "
                           "percentage. The display never goes above 99% or below 1%.")
            else:
                st.info("🔎 The model is very confident here, but no model can be 100% sure. Read this "
                        "as 'very high / very low'. The display never goes above 99% or below 1%.")
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

    total_eff = float(contrib["Contribution"].sum())
    score = INTERCEPT + total_eff
    with st.expander("🎓 How did the model calculate this? (simple explanation for students)"):
        st.markdown(f"""
Logistic Regression turns your inputs into a **score**, then turns the score into a **percentage**.

1. **Start** with the score of an *average* patient: `{INTERCEPT:+.2f}` (that is {BASELINE*100:.1f}% risk).
2. **Every input adds or subtracts points.** Inputs worse than average add points (red bars), better than average subtract points (green bars). This patient's inputs add up to `{total_eff:+.2f}`.
3. **Final score** = `{INTERCEPT:+.2f}` + `{total_eff:+.2f}` = **`{score:+.2f}`**
4. **Squash the score into 0–100%** with the S-shaped *sigmoid* curve: `risk = 1 / (1 + e^(−score))` = **{pct(prob)}**
        """)
        xs = np.linspace(-8, 8, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=1 / (1 + np.exp(-xs)) * 100, mode="lines", name="Sigmoid curve",
                                 line=dict(color=C_CYAN, width=4)))
        fig.add_trace(go.Scatter(x=[float(np.clip(score, -8, 8))], y=[prob * 100], mode="markers",
                                 name="This patient",
                                 marker=dict(size=16, color=vcolor, line=dict(color="#fff", width=2))))
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,.3)")
        show(layout(fig, 300, xaxis_title="Score (log-odds)", yaxis_title="Risk %",
                    yaxis=dict(range=[0, 100])))
        st.caption("A score of 0 means 50%. Each +1 on the score raises the odds about 2.7 times. "
                   "The curve flattens at both ends, so very high scores all look like ~100%.")

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
        f"<div class='ins info'>Start score <b>{INTERCEPT:+.2f}</b> + effects <b>{total:+.2f}</b> = "
        f"<b>{INTERCEPT + total:+.2f}</b> → predicted risk <b>{pct(prob)}</b> "
        f"(average patient: {BASELINE*100:.1f}%).</div>", unsafe_allow_html=True)

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
    tpl = pd.DataFrame([dict(zip(ALL_FEATURES, [38, 0, 2, 118, 190, 0, 0, 172, 0, 0.0, 1, 0, 3])),
                        dict(zip(ALL_FEATURES, [63, 1, 4, 160, 300, 1, 2, 118, 1, 3.2, 2, 3, 7]))])[FEATURES]
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
    terms = [pretty_term(t) for t in term_names()]
    cdf = pd.DataFrame({"Input": terms, "Coefficient": COEF,
                        "Odds ratio (per 1 SD)": np.exp(COEF)}).sort_values("Coefficient")
    fig = go.Figure(go.Bar(x=cdf["Coefficient"], y=cdf["Input"], orientation="h",
                           marker=dict(color=[C_HIGH if v > 0 else C_LOW for v in cdf["Coefficient"]])))
    show(layout(fig, max(430, 24 * len(cdf) + 80), xaxis_title="Standardised coefficient"))
    st.caption("Category inputs (chest pain, EKG, slope, Thallium) get one weight per label, because "
               "the labels are names, not amounts.")
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
<div class='card'><div class='kl'>The model in one line</div>
<p style='color:#cfdae7;margin-top:8px'>Inputs → (one-hot labels) → <code>StandardScaler</code> →
<code>LogisticRegression</code> → score → sigmoid → <b>probability of heart disease</b>.</p>
<div class='kl' style='margin-top:12px'>Risk bands</div>
<span class='chip ok'>&lt; 30% Low</span><span class='chip warn'>30–60% Moderate</span><span class='chip bad'>&gt; 60% High</span>
<p style='color:#8ea4bb;font-size:.85rem;margin-top:10px'>Bands were checked on 157,500 unseen patients: in the Low band about 9% really had disease, in Moderate about 44%, in High about 89%.</p>
</div>""", unsafe_allow_html=True)

    head("🤔 Questions students ask")
    with st.expander("Why does chest pain type 4 (Asymptomatic) give HIGH risk and type 1 (Typical angina) give LOW?"):
        st.markdown(f"""
It looks backwards, but the model is only copying the data. In this dataset (patients who were sent for heart tests):

| Chest pain type | Meaning | Share who had heart disease |
|---|---|---|
| 1 | Typical angina | **{RATE_CP[1]}%** |
| 2 | Atypical angina | **{RATE_CP[2]}%** |
| 3 | Non-anginal pain | **{RATE_CP[3]}%** |
| 4 | Asymptomatic (no classic chest pain) | **{RATE_CP[4]}%** |

**Why?** Many heart problems are *silent*. A patient with clear chest pain often has a harmless cause (muscle, stomach, stress), while a patient with *no* pain but abnormal test results is the one hiding a real problem. Also, the numbers 1–4 are **labels, not a scale**: 4 does not mean "4 times more pain". The model gives every label its own weight.
        """)
    with st.expander("Why do I get 'greater than 99%' when I set every input to its maximum?"):
        st.markdown("""
Maxing everything creates a patient who does not exist in real life: highest ST depression, 3 blocked vessels, reversible Thallium defect, angina, flat slope, and so on. Each of these alone already points strongly to disease in the data, and Logistic Regression **adds** all that evidence, so the score becomes huge and the sigmoid curve flattens near 100%.

That is the model being consistent, not a bug. Two safety rules are built in: the display never shows exactly 100% or 0%, and a *Reality check* warning appears for extreme combinations. A real doctor would never trust a "100%" either.
        """)
    with st.expander("What is Logistic Regression? (30-second version)"):
        st.markdown("""
Think of a **scoreboard**. Every input gives points: a risky value adds points, a healthy value subtracts points. The total score is converted into a percentage by an S-shaped curve (the *sigmoid*). The weights are learned from about 470,000 patient records so that the percentages match what really happened to those patients.
        """)
    with st.expander("📖 Glossary: what does every input mean?"):
        gloss = pd.DataFrame([
            ("Age", "Years of age", "Higher age → higher risk"),
            ("Sex", "Female / Male", "Males have a higher disease rate in this data"),
            ("Chest pain type", "Kind of chest pain (label 1–4)", "4 (asymptomatic) has the highest rate"),
            ("BP", "Resting blood pressure (mmHg)", "Tiny effect once other inputs are known"),
            ("Cholesterol", "Fat in blood (mg/dL)", "Small effect in this data"),
            ("FBS over 120", "Fasting blood sugar above 120?", "Tiny effect"),
            ("EKG results", "Resting heart electrical test", "LV hypertrophy → higher risk"),
            ("Max HR", "Highest pulse during exercise test", "Higher is healthier → lowers risk"),
            ("Exercise angina", "Chest pain during exercise?", "Yes → higher risk"),
            ("ST depression", "EKG dip during exercise", "Bigger dip → higher risk"),
            ("Slope of ST", "Direction of EKG ST segment", "Flat / down → higher risk than up"),
            ("Vessels (fluoroscopy)", "Blocked major vessels (0–3)", "More vessels → higher risk"),
            ("Thallium", "Nuclear stress scan of blood flow", "Defects (6, 7) → much higher risk"),
        ], columns=["Input", "What it is", "Effect on risk"])
        st.dataframe(gloss, hide_index=True, **STRETCH)

st.markdown(
    "<div class='disc'>⚠️ <b>Medical disclaimer:</b> CardioGuard AI is an educational machine-learning "
    "project. Its output is a statistical estimate, <b>not</b> a medical diagnosis. Always consult a "
    "qualified healthcare professional.</div>", unsafe_allow_html=True)
