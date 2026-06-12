import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict · Student ML",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Hide Streamlit chrome */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
header[data-testid="stHeader"]   { display: none !important; }
.stDeployButton { display: none !important; }
footer          { display: none !important; }

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(135deg, #fff0f6 0%, #fdf2f8 40%, #fce7f3 100%) !important;
    color: #500724 !important;
}

/* Main content — full width, space below navbar */
.block-container {
    padding: 0 2.5rem 3rem 2.5rem !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* ══════════════════════════════
   NAVBAR
══════════════════════════════ */
.navbar {
    display: flex;
    align-items: center;
    gap: 0;
    width: calc(100% + 5rem);
    margin: 0 -2.5rem 2.5rem -2.5rem;
    padding: 0 2.5rem;
    height: 68px;
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
    border-bottom: 1px solid rgba(236, 72, 153, 0.15);
    box-shadow: 0 1px 0 rgba(255,255,255,.9) inset,
                0 4px 24px rgba(190, 24, 93, 0.08);
    position: sticky;
    top: 0;
    z-index: 999;
}

/* Brand */
.nb-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-right: 10px;
    flex-shrink: 0;
    text-decoration: none !important;
}
.nb-brand-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #ec4899, #be185d);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 2px 8px rgba(190,24,93,.30);
    flex-shrink: 0;
}
.nb-brand-text {
    display: flex;
    flex-direction: column;
    gap: 1px;
}
.nb-brand-name {
    font-size: .95rem;
    font-weight: 800;
    color: #500724;
    letter-spacing: -.02em;
    line-height: 1;
}
.nb-brand-tag {
    font-size: .5rem;
    font-family: 'JetBrains Mono', monospace;
    color: #be185d;
    letter-spacing: .15em;
    text-transform: uppercase;
    opacity: .7;
}

/* Divider between brand and links */
.nb-sep {
    width: 1px;
    height: 28px;
    background: linear-gradient(180deg, transparent, #fbcfe8, transparent);
    margin: 0 20px;
    flex-shrink: 0;
}

/* Links wrapper */
.nb-links {
    display: flex;
    align-items: stretch;
    gap: 2px;
    height: 100%;
    flex: 1;
}

/* Each nav link */
.nb-link {
    position: relative;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 0 14px;
    height: 100%;
    color: #9d174d !important;
    font-size: .82rem;
    font-weight: 500;
    text-decoration: none !important;
    white-space: nowrap;
    transition: color .18s;
    border-radius: 0;
    font-family: 'Outfit', sans-serif;
    opacity: .65;
}
.nb-link:hover {
    opacity: 1;
    color: #be185d !important;
    text-decoration: none !important;
}
.nb-link:hover .nb-link-bg {
    opacity: 1;
}
/* Hover pill background */
.nb-link-bg {
    position: absolute;
    inset: 10px 4px;
    background: rgba(236, 72, 153, 0.07);
    border-radius: 8px;
    opacity: 0;
    transition: opacity .18s;
    pointer-events: none;
}
/* Active state */
.nb-link.active {
    opacity: 1;
    color: #be185d !important;
    font-weight: 700;
}
.nb-link.active .nb-link-bg {
    opacity: 1;
    background: rgba(236, 72, 153, 0.10);
}
/* Active bottom bar */
.nb-link.active::after {
    content: '';
    position: absolute;
    bottom: 0; left: 8px; right: 8px;
    height: 2.5px;
    background: linear-gradient(90deg, #ec4899, #be185d);
    border-radius: 2px 2px 0 0;
}
.nb-link-icon { font-size: .95rem; line-height: 1; position: relative; z-index: 1; }
.nb-link-label { position: relative; z-index: 1; }

/* Right side badge */
.nb-badge {
    margin-left: auto;
    flex-shrink: 0;
    background: linear-gradient(135deg, #fce7f3, #fbcfe8);
    border: 1px solid rgba(236,72,153,.2);
    border-radius: 999px;
    padding: 5px 14px;
    font-size: .68rem;
    font-family: 'JetBrains Mono', monospace;
    color: #be185d;
    letter-spacing: .08em;
    font-weight: 600;
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #fff5f9 100%) !important;
    border: 1px solid #fbcfe8 !important;
    border-top: 3px solid #ec4899 !important;
    border-radius: 16px !important;
    padding: 20px 22px !important;
    transition: transform .2s, box-shadow .2s !important;
    box-shadow: 0 2px 12px rgba(236,72,153,.10) !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(236,72,153,.22) !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #ec4899, #be185d) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
[data-testid="stMetricLabel"] { color: #9d174d !important; font-size: 0.75rem !important; letter-spacing: .06em !important; text-transform: uppercase !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(255,255,255,.75) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 4px !important;
    border: 1px solid #fbcfe8 !important;
    box-shadow: 0 1px 6px rgba(236,72,153,.08) !important;
    backdrop-filter: blur(6px) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #9d174d !important;
    border-radius: 9px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 9px 18px !important;
    transition: .15s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #ec4899, #be185d) !important;
    color: #fff !important;
    font-weight: 600 !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,.85) !important;
    border-color: #fbcfe8 !important;
    border-radius: 9px !important;
    color: #500724 !important;
}

/* Slider */
[data-testid="stSlider"] .stSlider { accent-color: #ec4899 !important; }

/* DataFrame */
[data-testid="stDataFrame"] {
    border: 1px solid #fbcfe8 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Form */
[data-testid="stForm"] {
    background: rgba(255,255,255,.80) !important;
    border: 1px solid #fbcfe8 !important;
    border-radius: 16px !important;
    padding: 28px !important;
    box-shadow: 0 2px 12px rgba(236,72,153,.08) !important;
    backdrop-filter: blur(6px) !important;
}

/* Button */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #f472b6 0%, #ec4899 50%, #be185d 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 12px 30px !important;
    letter-spacing: .03em !important;
    transition: opacity .2s, transform .2s, box-shadow .2s !important;
    box-shadow: 0 4px 16px rgba(236,72,153,.35) !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(236,72,153,.50) !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: #ec4899 !important; }

/* Divider */
hr { border-color: #fbcfe8 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #fdf2f8; }
::-webkit-scrollbar-thumb { background: #f9a8d4; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #ec4899; }

/* Success/Error alerts */
[data-testid="stAlert"] { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ──────────────────────────────────────────────
BG   = 'rgba(0,0,0,0)'
GRID = '#fce7f3'
TICK = '#fbcfe8'
FONT = '#9d174d'
COLORS = ['#ec4899','#f472b6','#be185d','#fb7185','#f9a8d4','#e879f9','#c084fc']
MODEL_COLORS = {
    'Random Forest':       '#ec4899',
    'Gradient Boosting':   '#be185d',
    'Logistic Regression': '#f472b6',
    'SVM':                 '#e879f9',
}

BASE_LAYOUT = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family='Outfit', color=FONT, size=12),
    margin=dict(l=10, r=10, t=44, b=10),
    xaxis=dict(gridcolor=GRID, linecolor=TICK, tickcolor=TICK, zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor=TICK, tickcolor=TICK, zeroline=False),
    legend=dict(bgcolor='rgba(255,240,246,.95)', bordercolor=TICK, borderwidth=1,
                font=dict(size=11)),
    hoverlabel=dict(bgcolor='#fff5f9', bordercolor=TICK, font_color='#831843'),
    colorway=COLORS,
)

def T(fig, **kw):
    layout = dict(BASE_LAYOUT)
    for k, v in kw.items():
        if k in layout and isinstance(layout[k], dict) and isinstance(v, dict):
            merged = dict(layout[k])
            merged.update(v)
            layout[k] = merged
        else:
            layout[k] = v
    fig.update_layout(**layout)
    return fig

# ── HELPERS ───────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.1):
    """Convert #rrggbb hex to rgba(r,g,b,alpha) string for Plotly."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def hdr(tag, title, desc=''):
    desc_html = f"<div style='font-size:.88rem;color:#9d174d;margin-top:7px;line-height:1.65'>{desc}</div>" if desc else ""
    html = (
        f'<div style="margin-bottom:28px;padding-bottom:20px;'
        f'border-bottom:2px solid transparent;'
        f'border-image:linear-gradient(90deg,#ec4899,#fbcfe8,transparent) 1">'
        f'<div style="font-size:.68rem;font-family:\'JetBrains Mono\',monospace;'
        f'background:linear-gradient(90deg,#ec4899,#be185d);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        f'letter-spacing:.16em;text-transform:uppercase;margin-bottom:7px">{tag}</div>'
        f'<div style="font-size:1.7rem;font-weight:800;color:#500724;line-height:1.15">{title}</div>'
        f'{desc_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def kard(content_html, accent='#ec4899', pad='22px 24px'):
    cleaned = "".join(line.strip() for line in content_html.splitlines())
    html = (
        f'<div style="background:rgba(255,255,255,.78);'
        f'backdrop-filter:blur(8px);'
        f'border:1px solid rgba(251,207,232,.8);border-left:4px solid {accent};'
        f'border-radius:14px;padding:{pad};margin-bottom:14px;'
        f'box-shadow:0 4px 16px rgba(236,72,153,.10);'
        f'transition:box-shadow .2s,transform .2s">'
        f'{cleaned}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def badge(text, color='#ec4899'):
    return f'<span style="background:rgba(37,99,235,.10);color:{color};font-size:.7rem;font-family:JetBrains Mono,monospace;padding:3px 10px;border-radius:999px;border:1px solid {color}44">{text}</span>'

# ── LOAD & TRAIN ──────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv('StudentPerformanceFactors.csv')
    df['Pass_Fail'] = (df['Exam_Score'] >= 70).astype(int)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)

    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df_enc = df.copy()
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col])

    feature_cols = [c for c in df_enc.columns if c not in ['Exam_Score','Pass_Fail']]
    X, y = df_enc[feature_cols], df_enc['Pass_Fail']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler    = StandardScaler()
    Xtr_sc    = scaler.fit_transform(X_train)
    Xte_sc    = scaler.transform(X_test)

    models = {
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM':                 SVC(probability=True, random_state=42),
    }
    SCALE = ['Logistic Regression','SVM']
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        Xtr = Xtr_sc if name in SCALE else X_train
        Xte = Xte_sc if name in SCALE else X_test
        Xcv = Xtr_sc if name in SCALE else X_train
        model.fit(Xtr, y_train)
        yp  = model.predict(Xte)
        ypr = model.predict_proba(Xte)[:,1]
        cvs = cross_val_score(model, Xcv, y_train, cv=cv, scoring='accuracy')
        fpr, tpr, _ = roc_curve(y_test, ypr)
        results[name] = dict(
            model=model, accuracy=accuracy_score(y_test, yp),
            auc=roc_auc_score(y_test, ypr), cv_mean=cvs.mean(), cv_std=cvs.std(),
            y_pred=yp, y_proba=ypr, fpr=fpr, tpr=tpr,
            cm=confusion_matrix(y_test, yp),
            report=classification_report(y_test, yp, output_dict=True),
        )

    rf_imp = pd.Series(models['Random Forest'].feature_importances_, index=feature_cols)\
               .sort_values(ascending=False).head(10)
    corr_v = df_enc[feature_cols+['Exam_Score']].corr()['Exam_Score']\
               .drop('Exam_Score').abs().sort_values(ascending=False).head(10)
    cat_ins = {}
    for col in ['Gender','School_Type','Parental_Involvement',
                'Motivation_Level','Family_Income','Teacher_Quality']:
        cat_ins[col] = df.groupby(col)['Exam_Score']\
                         .agg(['mean','count'])\
                         .rename(columns={'mean':'Rata-rata','count':'Jumlah'})\
                         .reset_index()
    return df, results, rf_imp, corr_v, cat_ins, feature_cols

# ── NAVBAR ────────────────────────────────────────────────────
PAGES = [
    "Overview",
    "Eksplorasi Data",
    "Model ML",
    "Evaluasi Model",
    "Feature Importance",
    "Insight Kategorikal",
    "Prediksi Siswa",
]

# ── NAVIGATION STATE ──────────────────────────────────────────
params = st.query_params
if "page" in params and params["page"] in PAGES:
    st.session_state.page = params["page"]
elif "page" not in st.session_state:
    st.session_state.page = "Overview"

page = st.session_state.page

# ── NAV RENDER ────────────────────────────────────────────────
ICONS = {
    "Overview":            "🏠",
    "Eksplorasi Data":     "🔍",
    "Model ML":            "🤖",
    "Evaluasi Model":      "📊",
    "Feature Importance":  "⭐",
    "Insight Kategorikal": "📂",
    "Prediksi Siswa":      "🔮",
}

sidebar_html = """
<div class="navbar">
  <a class="nb-brand" href="?page=Overview" target="_self">
    <div class="nb-brand-logo">🎓</div>
    <div class="nb-brand-text">
      <div class="nb-brand-name">EduPredict</div>
      <div class="nb-brand-tag">Student ML Dashboard</div>
    </div>
  </a>
  <div class="nb-sep"></div>
  <div class="nb-links">
"""
for label in PAGES:
    active = "active" if page == label else ""
    icon   = ICONS.get(label, "•")
    sidebar_html += f"""<a class="nb-link {active}" href="?page={label}" target="_self">
      <span class="nb-link-bg"></span>
      <span class="nb-link-icon">{icon}</span>
      <span class="nb-link-label">{label}</span>
    </a>"""

sidebar_html += f"""
  </div>
  <div class="nb-badge">ML · 2025</div>
</div>
"""

st.markdown(sidebar_html, unsafe_allow_html=True)


# ── LOAD ──────────────────────────────────────────────────────
with st.spinner("⏳ Melatih 4 model ML... (pertama kali ~30 detik)"):
    df, results, rf_imp, corr_v, cat_ins, feature_cols = load_and_train()

best = max(results, key=lambda m: results[m]['accuracy'])

# ══════════════════════════════════════════════════════════════
#  1. OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "Overview":
    hdr("01 — Overview", "Student Performance Dashboard",
        "Pipeline Machine Learning untuk memprediksi kelulusan siswa berdasarkan 19 faktor akademik & sosial.")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Siswa",    f"{len(df):,}")
    c2.metric("Siswa Lulus",    f"{df['Pass_Fail'].sum():,}",
              f"↑ {df['Pass_Fail'].mean()*100:.1f}% pass rate")
    c3.metric("Rata-rata Skor", f"{df['Exam_Score'].mean():.2f}",
              f"σ = {df['Exam_Score'].std():.2f}")
    c4.metric("Best Accuracy",  f"{results[best]['accuracy']*100:.2f}%", best)
    c5.metric("Best AUC-ROC",   f"{results[best]['auc']*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3,2], gap="large")

    with col1:
        score_dist = df['Exam_Score'].value_counts().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=score_dist.index, y=score_dist.values,
            marker=dict(
                color=['#e879f9' if s>=70 else '#f472b6' for s in score_dist.index],
                line=dict(width=0),
                opacity=0.85,
            ),
            hovertemplate="<b>Score %{x}</b><br>%{y} siswa<extra></extra>"
        ))
        fig.add_vline(x=69.5, line_dash="dash", line_color="#be185d", line_width=1.5,
                      annotation_text="Batas Lulus → 70",
                      annotation_font=dict(color="#be185d", size=11),
                      annotation_position="top left")
        T(fig, title="📊 Distribusi Nilai Ujian",
          xaxis_title="Nilai Ujian", yaxis_title="Jumlah Siswa",
          bargap=0.08)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pc = df['Pass_Fail'].sum()
        fc = len(df) - pc
        fig2 = go.Figure(go.Pie(
            labels=['Lulus (≥70)', 'Tidak Lulus (<70)'],
            values=[pc, fc], hole=.62,
            marker=dict(colors=['#e879f9','#f472b6'],
                        line=dict(color='#fff0f6', width=2)),
            textinfo='percent', textfont=dict(size=13),
            hovertemplate="%{label}<br><b>%{value:,}</b> siswa (%{percent})<extra></extra>"
        ))
        fig2.add_annotation(
            text=f"<b style='font-size:20px'>{df['Pass_Fail'].mean()*100:.1f}%</b><br><span style='font-size:11px;color:#be185d'>Pass Rate</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(color='#500724', size=14))
        T(fig2, title="🥧 Proporsi Kelulusan",
          legend=dict(orientation='h', yanchor='bottom', y=-0.18, xanchor='center', x=0.5))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:.68rem;font-family:'JetBrains Mono',monospace;
                color:#be185d;letter-spacing:.16em;text-transform:uppercase;
                margin-bottom:14px">Perbandingan Semua Model</div>""", unsafe_allow_html=True)

    fig3 = go.Figure()
    mnames = list(results.keys())
    bar_data = [
        ('accuracy','Accuracy','#ec4899'),
        ('auc','AUC-ROC','#f472b6'),
        ('cv_mean','CV Score','#e879f9'),
    ]
    for metric, label, color in bar_data:
        vals = [results[m][metric]*100 for m in mnames]
        fig3.add_trace(go.Bar(
            name=label, x=mnames, y=vals,
            marker=dict(color=color, opacity=0.85, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in vals],
            textposition='outside', textfont=dict(size=11),
        ))
    T(fig3, barmode='group', yaxis_range=[75,103],
      title="📈 Accuracy · AUC-ROC · CV Score", bargroupgap=0.1)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  2. EKSPLORASI DATA
# ══════════════════════════════════════════════════════════════
elif page == "Eksplorasi Data":
    hdr("02 — EDA", "Eksplorasi Data",
        "Distribusi fitur, pola antar variabel, dan kekuatan korelasi terhadap nilai ujian.")

    tab1, tab2, tab3 = st.tabs(["📋  Info Dataset", "📉  Distribusi & Scatter", "🔥  Korelasi"])

    # ── TAB 1: Info ──
    with tab1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            kard("""
            <div style="font-size:.68rem;font-family:'JetBrains Mono',monospace;
                        color:#be185d;letter-spacing:.12em;margin-bottom:14px">RINGKASAN DATASET</div>
            """ + "".join([
                f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #e2e8f4">'
                f'<span style="color:#9d174d;font-size:.83rem">{k}</span>'
                f'<span style="font-weight:600;font-size:.83rem;color:#0f172a">{v}</span></div>'
                for k,v in [
                    ("Jumlah Baris", f"{len(df):,}"),
                    ("Jumlah Kolom", str(len(df.columns))),
                    ("Missing Values (awal)", "235"),
                    ("Fitur Numerik", "7"),
                    ("Fitur Kategorikal", "12"),
                    ("Range Skor", f"{df['Exam_Score'].min()} – {df['Exam_Score'].max()}"),
                    ("Kelas Target", "Pass (≥70) / Fail (<70)"),
                ]
            ]))
        with c2:
            st.markdown("**Statistik Deskriptif**")
            st.dataframe(df.select_dtypes(include='number').describe().round(2),
                         use_container_width=True)

        st.markdown("<br>**Preview Dataset (10 baris pertama)**", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

    # ── TAB 2: Distribusi ──
    with tab2:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            num_col = st.selectbox("🔢 Pilih fitur numerik:",
                ['Exam_Score','Hours_Studied','Attendance','Sleep_Hours',
                 'Previous_Scores','Tutoring_Sessions','Physical_Activity'])
            fig = px.histogram(
                df, x=num_col, color='Pass_Fail',
                color_discrete_map={0:'#f472b6', 1:'#e879f9'},
                barmode='overlay', opacity=0.72, nbins=30,
                labels={'Pass_Fail':'Status','0':'Tidak Lulus','1':'Lulus'},
            )
            fig.for_each_trace(lambda t: t.update(
                name='Tidak Lulus' if t.name=='0' else 'Lulus'))
            T(fig, title=f"Distribusi — {num_col}",
              xaxis_title=num_col, yaxis_title="Frekuensi",
              legend_title_text="Status")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            cat_col = st.selectbox("📦 Boxplot per kategori:",
                ['Gender','School_Type','Motivation_Level','Family_Income',
                 'Parental_Involvement','Teacher_Quality'])
            fig2 = px.box(
                df, x=cat_col, y='Exam_Score', color=cat_col,
                color_discrete_sequence=COLORS,
                points=False,
            )
            fig2.add_hline(y=70, line_dash='dash', line_color='#f472b6',
                           annotation_text='Batas Lulus (70)',
                           annotation_font=dict(color='#f472b6', size=10))
            T(fig2, title=f"Exam Score per {cat_col}", showlegend=False,
              yaxis_title="Exam Score")
            st.plotly_chart(fig2, use_container_width=True)

        # Scatter — manual regression line (no statsmodels needed)
        st.markdown("---")
        c3, c4 = st.columns([1,3], gap="large")
        with c3:
            x_ax = st.selectbox("📌 Sumbu X (scatter):",
                ['Hours_Studied','Attendance','Previous_Scores',
                 'Sleep_Hours','Tutoring_Sessions','Physical_Activity'])
        with c4:
            pass

        # Compute manual OLS trendline
        x_vals = df[x_ax].values
        y_vals = df['Exam_Score'].values
        m_coef = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = np.polyval(m_coef, x_line)

        fig3 = go.Figure()
        for label, filt, color in [(1,'Lulus','#e879f9'), (0,'Tidak Lulus','#f472b6')]:
            mask = df['Pass_Fail'] == label
            fig3.add_trace(go.Scatter(
                x=df.loc[mask, x_ax], y=df.loc[mask, 'Exam_Score'],
                mode='markers', name=filt,
                marker=dict(color=color, size=4, opacity=0.45,
                            line=dict(width=0)),
                hovertemplate=f"<b>{x_ax}</b>: %{{x}}<br><b>Score</b>: %{{y}}<extra>{filt}</extra>"
            ))
        fig3.add_trace(go.Scatter(
            x=x_line, y=y_line, mode='lines', name='Trend (OLS)',
            line=dict(color='#fbbf24', width=2, dash='dot')
        ))
        T(fig3, title=f"Scatter: {x_ax} vs Exam Score",
          xaxis_title=x_ax, yaxis_title="Exam Score")
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 3: Korelasi ──
    with tab3:
        c1, c2 = st.columns([2,3], gap="large")
        with c1:
            kard("""
            <div style="font-size:.68rem;font-family:'JetBrains Mono',monospace;
                        color:#be185d;letter-spacing:.12em;margin-bottom:14px">
              TOP 10 KORELASI |r| VS EXAM_SCORE
            </div>
            """ + "".join([
                f'<div style="display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px solid #e2e8f4">'
                f'<span style="font-size:.7rem;font-family:JetBrains Mono,monospace;color:#cbd5e1;width:20px">#{i+1}</span>'
                f'<span style="font-size:.82rem;flex:1;color:#500724">{k.replace("_"," ")}</span>'
                f'<div style="width:80px;background:#e2e8f0;border-radius:4px;height:6px">'
                f'  <div style="width:{v/corr_v.max()*100:.0f}%;height:100%;border-radius:4px;'
                f'background:linear-gradient(90deg,#ec4899,#be185d)"></div></div>'
                f'<span style="font-size:.75rem;font-family:JetBrains Mono,monospace;color:#be185d;width:44px;text-align:right">{v:.3f}</span>'
                f'</div>'
                for i,(k,v) in enumerate(corr_v.items())
            ]))
        with c2:
            fig = px.bar(
                x=corr_v.values, y=corr_v.index,
                orientation='h',
                color=corr_v.values, color_continuous_scale='Blues',
                text=[f"{v:.3f}" for v in corr_v.values],
            )
            fig.update_traces(textposition='outside', textfont=dict(size=10),
                              marker_line_width=0)
            T(fig, title="Korelasi Absolut terhadap Exam Score",
              xaxis_title="|Korelasi Pearson|", yaxis_title="",
              coloraxis_showscale=False, yaxis_autorange='reversed')
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  3. MODEL ML
# ══════════════════════════════════════════════════════════════
elif page == "Model ML":
    hdr("03 — Models", "Perbandingan 4 Algoritma ML",
        "Evaluasi menyeluruh Random Forest, Gradient Boosting, Logistic Regression, dan SVM.")

    # Model cards
    cols = st.columns(4, gap="medium")
    for i, (name, r) in enumerate(results.items()):
        with cols[i]:
            is_best = name == best
            color   = MODEL_COLORS[name]
            kard(f"""
            <div style="font-size:.65rem;font-family:'JetBrains Mono',monospace;
                        color:{color};letter-spacing:.1em;margin-bottom:10px">
              {'⭐ BEST MODEL' if is_best else name.upper()}
            </div>
            <div style="font-size:1.05rem;font-weight:700;color:#500724;margin-bottom:16px">{name}</div>
            {''.join([
              f'<div style="margin-bottom:10px">'
              f'  <div style="display:flex;justify-content:space-between;margin-bottom:4px">'
              f'    <span style="font-size:.75rem;color:#64748b">{lbl}</span>'
              f'    <span style="font-size:.78rem;font-family:JetBrains Mono,monospace;color:{c}">{val}</span>'
              f'  </div>'
              f'  <div style="background:#fce7f3;border-radius:4px;height:5px">'
              f'    <div style="width:{pct}%;height:100%;border-radius:4px;background:{c}"></div>'
              f'  </div></div>'
              for lbl,val,c,pct in [
                ('Accuracy', f"{r['accuracy']*100:.2f}%", color, r['accuracy']*100),
                ('AUC-ROC',  f"{r['auc']*100:.2f}%",     '#f472b6', r['auc']*100),
                ('CV Score', f"{r['cv_mean']*100:.2f}%",  '#e879f9', r['cv_mean']*100),
              ]
            ])}
            <div style="font-size:.7rem;color:#fbcfe8;font-family:JetBrains Mono,monospace;margin-top:8px">
              CV Std ± {r['cv_std']*100:.2f}%
            </div>
            """, accent=color)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        fig = go.Figure()
        for metric, label, color in [
            ('accuracy','Accuracy','#ec4899'),
            ('auc','AUC-ROC','#f472b6'),
            ('cv_mean','CV Score','#e879f9'),
        ]:
            fig.add_trace(go.Bar(
                name=label, x=list(results.keys()),
                y=[results[m][metric]*100 for m in results],
                marker=dict(color=color, opacity=0.82, line=dict(width=0)),
                text=[f"{results[m][metric]*100:.1f}%" for m in results],
                textposition='outside', textfont=dict(size=10),
            ))
        T(fig, barmode='group', yaxis_range=[75,103],
          title="Metrik Perbandingan", bargroupgap=0.08)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cats = ['Accuracy','AUC-ROC','CV Mean']
        fig2 = go.Figure()
        for i, (name, r) in enumerate(results.items()):
            v = [r['accuracy']*100, r['auc']*100, r['cv_mean']*100]
            v_cl = v + [v[0]]
            fig2.add_trace(go.Scatterpolar(
                r=v_cl, theta=cats+[cats[0]],
                fill='toself', name=name,
                line=dict(color=MODEL_COLORS[name], width=2),
                fillcolor=hex_to_rgba(MODEL_COLORS[name], 0.12),
                opacity=0.85,
            ))
        fig2.update_layout(
            **{k:v for k,v in BASE_LAYOUT.items() if k not in ['xaxis','yaxis']},
            polar=dict(
                radialaxis=dict(visible=True, range=[80,100],
                                gridcolor=GRID, linecolor=TICK,
                                tickfont=dict(color='#fbcfe8', size=9)),
                bgcolor=BG,
                angularaxis=dict(linecolor=TICK, gridcolor=GRID),
            ),
            title="Radar Chart — Multi Metrik",
        )
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  4. EVALUASI
# ══════════════════════════════════════════════════════════════
elif page == "Evaluasi Model":
    hdr("04 — Evaluation", "Evaluasi Detail Model",
        "ROC Curve, Confusion Matrix, dan Classification Report per model.")

    model_sel = st.selectbox("Pilih Model:",
        list(results.keys()), index=list(results.keys()).index(best))
    r = results[model_sel]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",  f"{r['accuracy']*100:.2f}%")
    c2.metric("AUC-ROC",   f"{r['auc']*100:.2f}%")
    c3.metric("CV Mean",   f"{r['cv_mean']*100:.2f}%")
    c4.metric("CV Std",    f"±{r['cv_std']*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        color = MODEL_COLORS[model_sel]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r['fpr'], y=r['tpr'], mode='lines',
            name=f"{model_sel}  AUC={r['auc']*100:.2f}%",
            line=dict(color=color, width=2.5),
            fill='tozeroy',
            fillcolor=hex_to_rgba(color, 0.13),
        ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines', name='Baseline',
            line=dict(color='#fbcfe8', width=1.5, dash='dash')
        ))
        T(fig, title="ROC Curve",
          xaxis_title="False Positive Rate",
          yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cm    = r['cm']
        z     = cm.tolist()
        txt   = [[f"<b>{v}</b>" for v in row] for row in z]
        fig2  = go.Figure(go.Heatmap(
            z=z,
            x=['Pred: Tidak Lulus','Pred: Lulus'],
            y=['Actual: Tidak Lulus','Actual: Lulus'],
            colorscale=[[0,'#fff0f6'],[0.5,'#f9a8d4'],[1,'#be185d']],
            showscale=False,
            text=txt, texttemplate="%{text}",
            textfont=dict(size=20, color='#500724'),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        T(fig2, title="Confusion Matrix",
          xaxis=dict(side='bottom', gridcolor='rgba(0,0,0,0)', linecolor='rgba(0,0,0,0)'),
          yaxis=dict(autorange='reversed', gridcolor='rgba(0,0,0,0)', linecolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Classification Report**")
    rp = r['report']
    rows = []
    for cls, label in [('0','Tidak Lulus (0)'), ('1','Lulus (1)'), ('weighted avg','Weighted Avg')]:
        d = rp.get(cls, {})
        rows.append({
            'Kelas': label,
            'Precision': f"{d.get('precision',0)*100:.2f}%",
            'Recall':    f"{d.get('recall',0)*100:.2f}%",
            'F1-Score':  f"{d.get('f1-score',0)*100:.2f}%",
            'Support':   int(d.get('support',0)),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<br>**ROC Curve — Semua Model**", unsafe_allow_html=True)
    fig3 = go.Figure()
    for name, res in results.items():
        fig3.add_trace(go.Scatter(
            x=res['fpr'], y=res['tpr'], mode='lines',
            name=f"{name}  ({res['auc']*100:.1f}%)",
            line=dict(color=MODEL_COLORS[name], width=2),
        ))
    fig3.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Baseline',
                               line=dict(color='#fbcfe8',dash='dash',width=1.5)))
    T(fig3, title="ROC Curve — Perbandingan Semua Model",
      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  5. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    hdr("05 — Features", "Feature Importance — Random Forest",
        "Top 10 fitur paling berpengaruh terhadap prediksi kelulusan siswa.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        fi_sorted = rf_imp.sort_values()
        norm = fi_sorted / fi_sorted.max()
        fig.add_trace(go.Bar(
            x=fi_sorted.values, y=fi_sorted.index,
            orientation='h',
            marker=dict(
                color=fi_sorted.values,
                colorscale='Blues',
                line=dict(width=0),
                opacity=0.9,
            ),
            text=[f"{v*100:.2f}%" for v in fi_sorted.values],
            textposition='outside',
            textfont=dict(size=10),
        ))
        T(fig, title="Top 10 Feature Importance Score",
          xaxis_title="Importance Score", yaxis_title="",
          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure(go.Pie(
            labels=[l.replace('_',' ') for l in rf_imp.index],
            values=rf_imp.values,
            hole=0.5,
            marker=dict(colors=COLORS*2, line=dict(color='#fff0f6',width=2)),
            textinfo='label+percent',
            textfont=dict(size=10),
        ))
        T(fig2, title="Proporsi Feature Importance",
          legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Tabel Detail**")
    fi_df = pd.DataFrame({
        'Rank':    range(1, len(rf_imp)+1),
        'Fitur':   [f.replace('_',' ') for f in rf_imp.index],
        'Score':   rf_imp.values.round(5),
        'Persen':  [f"{v*100:.3f}%" for v in rf_imp.values],
    })
    st.dataframe(fi_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
#  6. CATEGORICAL INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "Insight Kategorikal":
    hdr("06 — Insights", "Insight Kategorikal",
        "Rata-rata nilai ujian dan distribusi siswa berdasarkan faktor sosial & demografis.")

    cat_sel = st.selectbox("Pilih Kategori:", list(cat_ins.keys()))
    data    = cat_ins[cat_sel].sort_values('Rata-rata', ascending=False)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        colors_bar = [COLORS[i % len(COLORS)] for i in range(len(data))]
        fig.add_trace(go.Bar(
            x=data[cat_sel], y=data['Rata-rata'],
            marker=dict(color=colors_bar, opacity=0.85, line=dict(width=0)),
            text=data['Rata-rata'].round(2), textposition='outside',
            textfont=dict(size=11),
        ))
        fig.add_hline(y=70, line_dash='dash', line_color='#f472b6', line_width=1.5,
                      annotation_text='Batas Lulus (70)',
                      annotation_font=dict(color='#f472b6', size=10))
        T(fig, title=f"Rata-rata Nilai — {cat_sel}",
          yaxis_range=[60,76], yaxis_title="Rata-rata Exam Score")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=data[cat_sel], y=data['Jumlah'],
            marker=dict(color=colors_bar, opacity=0.75, line=dict(width=0)),
            text=data['Jumlah'], textposition='outside', textfont=dict(size=11),
        ))
        T(fig2, title=f"Jumlah Siswa — {cat_sel}", yaxis_title="Jumlah Siswa")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Detail Tabel**")
    st.dataframe(data.round(2), use_container_width=True, hide_index=True)

    st.markdown("<br>**Semua Kategori Sekaligus**", unsafe_allow_html=True)
    fig3 = make_subplots(rows=2, cols=3,
        subplot_titles=[c.replace('_',' ') for c in cat_ins.keys()],
        vertical_spacing=0.22, horizontal_spacing=0.07)
    for idx, (col_name, d) in enumerate(cat_ins.items()):
        row, col = idx//3+1, idx%3+1
        ds = d.sort_values('Rata-rata', ascending=False)
        fig3.add_trace(go.Bar(
            x=ds[col_name], y=ds['Rata-rata'],
            marker=dict(color=COLORS[idx%len(COLORS)], opacity=0.82, line=dict(width=0)),
            showlegend=False,
            text=ds['Rata-rata'].round(1), textposition='outside',
            textfont=dict(size=9),
        ), row=row, col=col)
        fig3.add_hline(y=70, line_dash='dot', line_color='rgba(190,24,93,0.25)',
                       line_width=1, row=row, col=col)
    fig3.update_layout(
        height=540, **{k:v for k,v in BASE_LAYOUT.items()
                        if k not in ['xaxis','yaxis','margin']},
        margin=dict(l=10,r=10,t=60,b=10),
    )
    fig3.update_annotations(font=dict(color='#9d174d', size=11))
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  7. PREDIKSI SISWA
# ══════════════════════════════════════════════════════════════
elif page == "Prediksi Siswa":
    hdr("07 — Predict", "Prediksi Kelulusan Siswa Baru",
        "Masukkan profil siswa untuk mendapatkan prediksi kelulusan secara real-time dari semua model.")

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("**📚 Akademik**")
            hours      = st.slider("Jam Belajar / Minggu", 1, 44, 20)
            attendance = st.slider("Kehadiran (%)", 60, 100, 80)
            prev_score = st.slider("Nilai Sebelumnya", 50, 100, 75)
            tutoring   = st.slider("Sesi Tutoring / Bulan", 0, 8, 1)
        with c2:
            st.markdown("**🏠 Personal**")
            sleep_h    = st.slider("Jam Tidur / Hari", 4, 10, 7)
            physical   = st.slider("Aktivitas Fisik (jam/minggu)", 0, 6, 3)
            gender     = st.selectbox("Gender", ['Male','Female'])
            disability = st.selectbox("Learning Disability", ['No','Yes'])
            extra      = st.selectbox("Ekstrakurikuler", ['Yes','No'])
        with c3:
            st.markdown("**🌐 Lingkungan**")
            parental   = st.selectbox("Keterlibatan Orang Tua", ['Low','Medium','High'])
            resources  = st.selectbox("Akses Sumber Belajar", ['Low','Medium','High'])
            motivation = st.selectbox("Motivasi", ['Low','Medium','High'])
            internet   = st.selectbox("Akses Internet", ['Yes','No'])
            income     = st.selectbox("Pendapatan Keluarga", ['Low','Medium','High'])
            teacher_q  = st.selectbox("Kualitas Guru", ['Low','Medium','High'])
            school_t   = st.selectbox("Jenis Sekolah", ['Public','Private'])
            peer       = st.selectbox("Pengaruh Teman", ['Positive','Neutral','Negative'])
            par_edu    = st.selectbox("Pendidikan Orang Tua",
                ['High School','College','Postgraduate'])
            distance   = st.selectbox("Jarak dari Rumah", ['Near','Moderate','Far'])

        submitted = st.form_submit_button("🔮  Prediksi Kelulusan", use_container_width=True)

    if submitted:
        lmh = {'Low':0,'Medium':1,'High':2}
        inp_dict = {
            'Hours_Studied': hours, 'Attendance': attendance,
            'Parental_Involvement': lmh[parental],
            'Access_to_Resources': lmh[resources],
            'Extracurricular_Activities': 1 if extra=='Yes' else 0,
            'Sleep_Hours': sleep_h, 'Previous_Scores': prev_score,
            'Motivation_Level': lmh[motivation],
            'Internet_Access': 1 if internet=='Yes' else 0,
            'Tutoring_Sessions': tutoring, 'Family_Income': lmh[income],
            'Teacher_Quality': lmh[teacher_q],
            'School_Type': 1 if school_t=='Public' else 0,
            'Peer_Influence': {'Positive':2,'Neutral':1,'Negative':0}[peer],
            'Physical_Activity': physical,
            'Learning_Disabilities': 1 if disability=='Yes' else 0,
            'Parental_Education_Level': {'High School':0,'College':1,'Postgraduate':2}[par_edu],
            'Distance_from_Home': {'Near':0,'Moderate':1,'Far':2}[distance],
            'Gender': 1 if gender=='Male' else 0,
        }
        inp = pd.DataFrame([inp_dict])[feature_cols]

        st.markdown("<br>", unsafe_allow_html=True)

        # 2 model result cards — Random Forest & Gradient Boosting only
        display_models = ['Random Forest', 'Gradient Boosting']
        _, c_left, c_right, _ = st.columns([1, 2, 2, 1], gap="medium")
        for i, name in enumerate(display_models):
            res   = results[name]
            pred  = res['model'].predict(inp)[0]
            prob  = res['model'].predict_proba(inp)[0]
            lulus = pred == 1
            clr   = '#be185d' if lulus else '#f472b6'
            icon  = '✅' if lulus else '❌'
            col   = c_left if i == 0 else c_right
            with col:
                kard(f"""
                <div style="font-size:.6rem;font-family:'JetBrains Mono',monospace;
                            color:{MODEL_COLORS[name]};letter-spacing:.1em;margin-bottom:8px">
                  {name.upper()}
                </div>
                <div style="font-size:2.8rem;line-height:1">{icon}</div>
                <div style="font-size:1.25rem;font-weight:800;color:{clr};margin:10px 0 6px">
                  {'LULUS' if lulus else 'TIDAK LULUS'}
                </div>
                <div style="font-size:.82rem;color:#9d174d">Prob. Lulus:</div>
                <div style="font-size:1.6rem;font-weight:800;color:{clr};
                            font-family:'JetBrains Mono',monospace">{prob[1]*100:.1f}%</div>
                <div style="background:#fce7f3;border-radius:4px;height:6px;margin-top:12px">
                  <div style="width:{prob[1]*100:.0f}%;height:100%;border-radius:4px;
                              background:linear-gradient(90deg,#f9a8d4,{clr})"></div>
                </div>
                """, accent=clr)

        # Gauge best model
        st.markdown("<br>", unsafe_allow_html=True)
        prob_best = results[best]['model'].predict_proba(inp)[0][1]
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_best*100,
            title={'text': f"Probabilitas Lulus — <b>{best}</b> (Best Model)",
                   'font': {'color':'#500724','size':15}},
            number={'suffix':'%','font':{'color':'#ec4899','size':44}},
            delta={'reference':50, 'suffix':'%',
                   'increasing':{'color':'#e879f9'},
                   'decreasing':{'color':'#be185d'}},
            gauge={
                'axis': {'range':[0,100], 'tickfont':{'color':'#be185d','size':10},
                         'tickcolor':'#cbd5e1'},
                'bar':  {'color':'#ec4899','thickness':0.28},
                'bgcolor': BG,
                'borderwidth': 0,
                'steps': [
                    {'range':[0,40],  'color':'rgba(236,72,153,.08)'},
                    {'range':[40,60], 'color':'rgba(251,207,232,.15)'},
                    {'range':[60,100],'color':'rgba(190,24,93,.08)'},
                ],
                'threshold': {'line':{'color':'#f472b6','width':3}, 'value':50},
            }
        ))
        fig_g.update_layout(
            paper_bgcolor=BG, font_color='#9d174d',
            height=280, margin=dict(l=40,r=40,t=60,b=10)
        )
        st.plotly_chart(fig_g, use_container_width=True)

        verdict = "✅ Siswa ini diprediksi **LULUS**" if prob_best>=0.5 \
                  else "❌ Siswa ini diprediksi **TIDAK LULUS**"
        color_v = "#be185d" if prob_best>=0.5 else "#f472b6"
        st.markdown(f"""
        <div style="background:rgba(0,0,0,.2);border:1px solid {color_v}44;
                    border-left:4px solid {color_v};border-radius:12px;
                    padding:18px 22px;margin-top:8px;text-align:center;
                    font-size:1.05rem;font-weight:600;color:{color_v}">
          {verdict} &nbsp;·&nbsp; Probabilitas {prob_best*100:.1f}%
        </div>""", unsafe_allow_html=True)