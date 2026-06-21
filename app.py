import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io

st.set_page_config(
    page_title="HELP International – Aid Prioritisation",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (matches the HTML theme) ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --ink:      #10151B;
  --ink2:     #171D25;
  --ink3:     #1E2630;
  --paper:    #F6F3EC;
  --paper2:   #EAE6DC;
  --wire:     #5B6470;
  --wire-soft:#38414C;
  --clay:     #C1502E;
  --clay-s:   rgba(193,80,46,0.16);
  --amber:    #D9A02C;
  --amber-s:  rgba(217,160,44,0.16);
  --teal:     #3E7A5E;
  --teal-s:   rgba(62,122,94,0.16);
  --display:  'Source Serif 4', Georgia, serif;
  --body:     'Inter', system-ui, sans-serif;
  --mono:     'JetBrains Mono', ui-monospace, monospace;
}

html, body, [class*="css"] {
  font-family: var(--body) !important;
  background: var(--ink) !important;
  color: var(--paper) !important;
}

.stApp { background: var(--ink); }
.block-container { padding: 0 2.2rem 4rem; max-width: 100%; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--ink2); }
::-webkit-scrollbar-thumb { background: var(--wire-soft); border-radius: 3px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--ink2);
  border-radius: 2px;
  padding: 4px 6px;
  gap: 2px;
  border: 1px solid var(--wire-soft);
  margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 2px;
  color: var(--wire);
  font-size: 0.72rem;
  font-family: var(--mono);
  font-weight: 400;
  padding: 8px 16px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: none;
}
.stTabs [aria-selected="true"] {
  background: var(--amber-s) !important;
  color: var(--amber) !important;
  border: 1px solid var(--amber) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ── Masthead ── */
.masthead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 16px;
  margin-bottom: 28px;
  border-bottom: 1px solid var(--wire-soft);
  flex-wrap: wrap;
  gap: 10px;
}
.org-mark {
  font-family: var(--display);
  font-weight: 700;
  font-size: 1.1rem;
  color: var(--paper);
  letter-spacing: 0.01em;
}
.org-mark-sub { font-weight: 400; color: var(--wire); font-style: italic; }
.report-tag {
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  color: var(--wire);
  text-transform: uppercase;
}

/* ── Hero ── */
.hero-title {
  font-family: var(--display);
  font-weight: 600;
  font-size: clamp(2rem,4.5vw,3.2rem);
  line-height: 1.1;
  color: var(--paper);
  margin: 0 0 12px;
  letter-spacing: -0.01em;
}
.hero-copy {
  font-size: 0.92rem;
  line-height: 1.65;
  color: #C7C2B4;
  max-width: 660px;
  margin: 0 0 32px;
}

/* ── KPI strip ── */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(4,1fr);
  gap: 12px;
  margin-bottom: 32px;
}
.kpi-card {
  background: var(--ink2);
  border: 1px solid var(--wire-soft);
  border-radius: 4px;
  padding: 18px 16px;
  transition: border-color .2s;
}
.kpi-card:hover { border-color: var(--amber); }
.kpi-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--wire);
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 10px;
}
.kpi-dot { width:6px;height:6px;border-radius:50%;display:inline-block;flex-shrink:0; }
.kpi-val {
  font-family: var(--display);
  font-weight: 600;
  font-size: 1.9rem;
  color: var(--paper);
  line-height: 1;
  margin-bottom: 4px;
}
.kpi-sub { font-size: 0.72rem; color: var(--wire); }

/* ── Panel ── */
.panel {
  background: var(--ink2);
  border: 1px solid var(--wire-soft);
  border-radius: 4px;
  padding: 22px;
  margin-bottom: 16px;
}
.panel-head {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 18px;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--wire-soft);
  padding-bottom: 12px;
}
.panel-num {
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.12em;
  color: var(--wire);
  border: 1px solid var(--wire-soft);
  padding: 2px 7px;
  border-radius: 2px;
}
.panel-head h2 {
  font-family: var(--display);
  font-weight: 600;
  font-size: 1.15rem;
  color: var(--paper);
  margin: 0;
}

/* ── Section label ── */
.sec-label {
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--wire);
  display: flex;
  align-items: center;
  gap: 7px;
  margin: 0 0 14px;
}
.sec-swatch { width:7px;height:7px;border-radius:50%;display:inline-block; }

/* ── Cluster cards ── */
.cluster-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin-bottom: 24px; }
.cluster-card {
  border: 1px solid var(--wire-soft);
  border-radius: 4px;
  padding: 20px 18px;
  transition: border-color .2s;
}
.cluster-card.c-underdeveloped { background: var(--clay-s); border-color: var(--clay); }
.cluster-card.c-developing     { background: var(--amber-s); border-color: var(--amber); }
.cluster-card.c-developed      { background: var(--teal-s); border-color: var(--teal); }
.c-badge {
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--wire);
  margin-bottom: 10px;
  display: block;
}
.c-name {
  font-family: var(--display);
  font-weight: 700;
  font-size: 1.3rem;
  color: var(--paper);
  margin-bottom: 2px;
}
.c-count { font-size: 2.6rem; font-family: var(--display); font-weight: 600; line-height: 1; margin-bottom: 2px; }
.c-underdeveloped .c-count { color: var(--clay); }
.c-developing .c-count { color: var(--amber); }
.c-developed .c-count { color: var(--teal); }
.c-sub { font-size: 0.72rem; color: var(--wire); text-transform: uppercase; font-family: var(--mono); letter-spacing: 0.06em; margin-bottom: 16px; }
.c-metrics { display: flex; flex-direction: column; gap: 8px; border-top: 1px dashed var(--wire-soft); padding-top: 14px; }
.c-metric { display: flex; justify-content: space-between; }
.c-metric-label { font-size: 0.76rem; color: var(--wire); }
.c-metric-val { font-family: var(--mono); font-size: 0.8rem; color: var(--paper); }

/* ── Priority rows ── */
.prow {
  display: flex;
  align-items: center;
  padding: 9px 12px;
  border-radius: 2px;
  margin-bottom: 4px;
  background: var(--ink2);
  border: 1px solid var(--wire-soft);
  transition: border-color .2s;
}
.prow:hover { border-color: var(--amber); }
.prow-rank { font-family: var(--mono); font-size: 0.68rem; color: var(--clay); width: 30px; }
.prow-country { font-size: 0.88rem; font-weight: 500; color: var(--paper); flex: 1; }
.prow-track { flex: 2; height: 5px; background: var(--wire-soft); border-radius: 999px; margin: 0 12px; overflow: hidden; }
.prow-fill { height: 100%; border-radius: 999px; background: var(--clay); }
.prow-score { font-family: var(--mono); font-size: 0.72rem; color: var(--amber); width: 44px; text-align: right; }

/* ── Insight cards ── */
.insight-card {
  background: var(--ink2);
  border: 1px solid var(--wire-soft);
  border-left: 3px solid var(--clay);
  border-radius: 2px;
  padding: 14px 16px;
  margin-bottom: 10px;
  transition: border-color .2s;
}
.insight-card:hover { border-left-color: var(--amber); }
.insight-tag {
  font-family: var(--mono);
  font-size: 0.6rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--wire);
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.insight-text { font-size: 0.87rem; color: #C7C2B4; line-height: 1.65; }

/* ── Alloc rows ── */
.arow {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid var(--wire-soft);
}
.arow-country { font-size: 0.84rem; color: var(--paper); width: 170px; }
.arow-track { flex: 1; height: 6px; background: var(--wire-soft); border-radius: 999px; overflow: hidden; }
.arow-fill { height: 100%; border-radius: 999px; background: var(--clay); }
.arow-pct { font-family: var(--mono); font-size: 0.68rem; color: var(--wire); width: 44px; text-align: right; }
.arow-amt { font-family: var(--mono); font-size: 0.76rem; color: var(--amber); width: 80px; text-align: right; }

/* ── Divider ── */
.wire-div { height: 1px; background: var(--wire-soft); margin: 24px 0; }

/* ── Metric overrides ── */
[data-testid="metric-container"] {
  background: var(--ink2);
  border: 1px solid var(--wire-soft);
  border-radius: 4px;
  padding: 14px 16px;
}
[data-testid="metric-container"] label { color: var(--wire) !important; font-family: var(--mono) !important; font-size: 0.65rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--paper) !important; font-family: var(--display) !important; font-size: 1.7rem !important; font-weight: 600 !important; }

/* ── Plotly container ── */
.plot-wrap { background: var(--ink2); border: 1px solid var(--wire-soft); border-radius: 4px; padding: 4px; margin-bottom: 14px; }

/* ── Select/input ── */
.stSelectbox > div > div, .stTextInput > div > div > input {
  background: var(--ink2) !important;
  border-color: var(--wire-soft) !important;
  color: var(--paper) !important;
  font-family: var(--body) !important;
  border-radius: 2px !important;
}
.stDownloadButton > button {
  background: transparent !important;
  border: 1px solid var(--amber) !important;
  color: var(--amber) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.08em !important;
  border-radius: 999px !important;
  padding: 8px 20px !important;
}
.stDownloadButton > button:hover { background: var(--amber) !important; color: var(--ink) !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ─────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor='#171D25',
    plot_bgcolor='#171D25',
    font=dict(family='Inter', color='#5B6470', size=11),
    xaxis=dict(gridcolor='#38414C', zerolinecolor='#38414C', tickfont=dict(color='#5B6470')),
    yaxis=dict(gridcolor='#38414C', zerolinecolor='#38414C', tickfont=dict(color='#5B6470')),
    legend=dict(bgcolor='#10151B', bordercolor='#38414C', borderwidth=1, font=dict(color='#C7C2B4')),
    margin=dict(l=20, r=20, t=36, b=20),
)
CMAP = {'Underdeveloped':'#C1502E', 'Developing':'#D9A02C', 'Developed':'#3E7A5E'}

# ── Data & ML ────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("Dataset_8.csv")
    X = df.drop('country', axis=1)
    cols = X.columns.tolist()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca2 = PCA(n_components=2); Xp = pca2.fit_transform(Xs)
    pca_full = PCA(); pca_full.fit(Xs)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = km.fit_predict(Xs)
    df['PC1'] = Xp[:,0]; df['PC2'] = Xp[:,1]
    # label clusters by GDP
    gmap = df.groupby('Cluster')['gdpp'].mean()
    crit  = gmap.idxmin()
    dev   = gmap.idxmax()
    mid   = [c for c in [0,1,2] if c!=crit and c!=dev][0]
    role_map = {crit:'Underdeveloped', mid:'Developing', dev:'Developed'}
    df['ClusterRole'] = df['Cluster'].map(role_map)
    df['Need_Score'] = df['child_mort']*0.4 + (1/df['income'].clip(lower=1))*100 + df['total_fer']*0.3
    v1, v2 = pca2.explained_variance_ratio_
    return df, cols, v1, v2, pca_full

df, feat_cols, v1, v2, pca_full = load()
top10 = df[df['ClusterRole']=='Underdeveloped'].sort_values('Need_Score', ascending=False).head(10)
summary = df.groupby('ClusterRole').agg(
    child_mort=('child_mort','mean'), income=('income','mean'),
    gdpp=('gdpp','mean'), life_expec=('life_expec','mean'),
    total_fer=('total_fer','mean'), count=('country','count')
).reset_index()
def srow(role): return summary[summary['ClusterRole']==role].iloc[0]

# ── Masthead ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
  <div><span class="org-mark">HELP <span class="org-mark-sub">International</span></span></div>
  <span class="report-tag">AI Humanitarian Decision Intelligence</span>
</div>
<h1 class="hero-title">Country Segmentation<br>for Aid Prioritisation</h1>
<p class="hero-copy">Principal Component Analysis and K-Means Clustering applied to 167 countries
across 9 socio-economic indicators — providing executive-level intelligence for humanitarian fund allocation.</p>
""", unsafe_allow_html=True)

# KPI strip
ud = srow('Underdeveloped')
st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-card">
    <div class="kpi-label"><span class="kpi-dot" style="background:#3E7A5E"></span>Countries Analysed</div>
    <div class="kpi-val">167</div>
    <div class="kpi-sub">Full global dataset</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label"><span class="kpi-dot" style="background:#C1502E"></span>Critical Need</div>
    <div class="kpi-val">{int(ud['count'])}</div>
    <div class="kpi-sub">Underdeveloped nations</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label"><span class="kpi-dot" style="background:#D9A02C"></span>Avg Child Mortality</div>
    <div class="kpi-val">{df['child_mort'].mean():.1f}</div>
    <div class="kpi-sub">per 1,000 live births</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label"><span class="kpi-dot" style="background:#3E7A5E"></span>Variance Retained</div>
    <div class="kpi-val">63.13%</div>
    <div class="kpi-sub">2 principal components</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(["01 Overview","02 PCA","03 Clusters","04 Priority Index","05 AI Insights","06 Fund Simulator","07 Analytics"])

# ════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">GLOBAL</span><h2>3D Country Intelligence Map</h2></div>', unsafe_allow_html=True)
    fig3d = px.scatter_3d(df, x='income', y='life_expec', z='child_mort',
        color='ClusterRole', hover_name='country',
        color_discrete_map=CMAP, opacity=0.85,
        labels={'income':'Income','life_expec':'Life Expectancy','child_mort':'Child Mortality','ClusterRole':'Cluster'})
    fig3d.update_traces(marker=dict(size=5, line=dict(width=0.4, color='#10151B')))
    fig3d.update_layout(paper_bgcolor='#171D25',
        scene=dict(bgcolor='#10151B',
            xaxis=dict(backgroundcolor='#10151B',gridcolor='#38414C',color='#5B6470'),
            yaxis=dict(backgroundcolor='#10151B',gridcolor='#38414C',color='#5B6470'),
            zaxis=dict(backgroundcolor='#10151B',gridcolor='#38414C',color='#5B6470')),
        font=dict(family='Inter',color='#5B6470'),
        legend=dict(bgcolor='#10151B',bordercolor='#38414C',borderwidth=1,font=dict(color='#C7C2B4')),
        height=520, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">CORRELATION</span><h2>Feature Heatmap</h2></div>', unsafe_allow_html=True)
        corr = df[feat_cols].corr()
        fig_h = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale=[[0,'#C1502E'],[0.5,'#171D25'],[1,'#3E7A5E']],
            zmin=-1, zmax=1, text=np.round(corr.values,2), texttemplate='%{text}', textfont=dict(size=8)))
        fig_h.update_layout(**PL, height=360)
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">DISTRIBUTION</span><h2>Cluster Breakdown</h2></div>', unsafe_allow_html=True)
        counts = df['ClusterRole'].value_counts()
        fig_pie = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.58,
            marker=dict(colors=['#C1502E','#D9A02C','#3E7A5E'], line=dict(color='#10151B',width=2)),
            textfont=dict(color='#F6F3EC',size=11)))
        fig_pie.update_layout(**PL, height=360,
            annotations=[dict(text='167<br><span style="font-size:11px">Nations</span>',x=0.5,y=0.5,font_size=20,font_color='#F6F3EC',showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — PCA
# ════════════════════════════════════════════════════════
with tabs[1]:
    c1,c2,c3 = st.columns(3)
    for col, label, val, sub in [
        (c1,"PC1 Variance",f"{v1*100:.2f}%","Economic Development Dimension"),
        (c2,"PC2 Variance",f"{v2*100:.2f}%","Trade Characteristics Dimension"),
        (c3,"Total Retained",f"{(v1+v2)*100:.2f}%","2 principal components"),
    ]:
        with col:
            st.markdown(f'<div class="panel" style="text-align:center;padding:20px"><div class="kpi-label" style="justify-content:center">{label}</div><div class="kpi-val" style="font-size:2rem;color:#D9A02C">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    ca, cb = st.columns([3,2])
    with ca:
        st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">PROJECTION</span><h2>PCA Country Space</h2></div>', unsafe_allow_html=True)
        fig_pca = px.scatter(df, x='PC1', y='PC2', color='ClusterRole', hover_name='country',
            color_discrete_map=CMAP, opacity=0.85,
            labels={'PC1':f'PC1 ({v1*100:.1f}% var)','PC2':f'PC2 ({v2*100:.1f}% var)','ClusterRole':'Cluster'})
        fig_pca.update_traces(marker=dict(size=7, line=dict(width=0.4,color='#10151B')))
        fig_pca.update_layout(**PL, height=420)
        st.plotly_chart(fig_pca, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">INTERPRETATION</span><h2>Dimensions</h2></div>', unsafe_allow_html=True)
        for icon,tag,text in [
            ("●","PC1 — Economic Development",f"Captures {v1*100:.1f}% of variance. High positive scores reflect strong income, GDP, and life expectancy. Negative scores indicate underdevelopment and elevated child mortality."),
            ("●","PC2 — Trade Characteristics",f"Captures {v2*100:.1f}% of variance. Differentiates trade-surplus economies from import-dependent nations based on export/import dynamics."),
            ("●","Selection Rationale",f"Two components selected via Scree Plot elbow, retaining 63.13% of total variance — sufficient for meaningful cluster separation from 9 original features."),
        ]:
            st.markdown(f'<div class="insight-card"><div class="insight-tag"><span style="color:#D9A02C">{icon}</span>{tag}</div><div class="insight-text">{text}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">VARIANCE</span><h2>Scree Plot & Cumulative Variance</h2></div>', unsafe_allow_html=True)
    n = len(pca_full.explained_variance_ratio_)
    fig_sc = make_subplots(rows=1,cols=2,subplot_titles=['Individual Explained Variance','Cumulative Explained Variance'])
    fig_sc.add_trace(go.Scatter(x=list(range(1,n+1)),y=pca_full.explained_variance_ratio_*100,
        mode='lines+markers',line=dict(color='#C1502E',width=2),
        marker=dict(size=7,color='#C1502E',line=dict(color='#10151B',width=1.5)),name='Individual'),row=1,col=1)
    fig_sc.add_trace(go.Scatter(x=list(range(1,n+1)),y=np.cumsum(pca_full.explained_variance_ratio_)*100,
        mode='lines+markers',line=dict(color='#3E7A5E',width=2),
        marker=dict(size=7,color='#3E7A5E',line=dict(color='#10151B',width=1.5)),name='Cumulative'),row=1,col=2)
    fig_sc.add_hline(y=63.13,line_dash='dash',line_color='rgba(193,80,46,0.4)',row=1,col=2)
    fig_sc.update_layout(**PL,height=340)
    fig_sc.update_xaxes(gridcolor='#38414C',tickfont=dict(color='#5B6470'))
    fig_sc.update_yaxes(gridcolor='#38414C',tickfont=dict(color='#5B6470'))
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3 — CLUSTERS
# ════════════════════════════════════════════════════════
with tabs[2]:
    ud=srow('Underdeveloped'); dv=srow('Developing'); dd=srow('Developed')
    st.markdown(f"""
    <div class="cluster-grid">
      <div class="cluster-card c-underdeveloped">
        <span class="c-badge">Cluster · Critical Priority</span>
        <div class="c-name">Underdeveloped</div>
        <div class="c-count">{int(ud['count'])}</div>
        <div class="c-sub">Nations · Urgent Aid</div>
        <div class="c-metrics">
          <div class="c-metric"><span class="c-metric-label">Child Mortality</span><span class="c-metric-val">{ud['child_mort']:.1f}/1K</span></div>
          <div class="c-metric"><span class="c-metric-label">Avg Income</span><span class="c-metric-val">${ud['income']:,.0f}</span></div>
          <div class="c-metric"><span class="c-metric-label">GDP per Capita</span><span class="c-metric-val">${ud['gdpp']:,.0f}</span></div>
          <div class="c-metric"><span class="c-metric-label">Life Expectancy</span><span class="c-metric-val">{ud['life_expec']:.1f} yrs</span></div>
          <div class="c-metric"><span class="c-metric-label">Fertility Rate</span><span class="c-metric-val">{ud['total_fer']:.2f}</span></div>
        </div>
      </div>
      <div class="cluster-card c-developing">
        <span class="c-badge">Cluster · Medium Priority</span>
        <div class="c-name">Developing</div>
        <div class="c-count">{int(dv['count'])}</div>
        <div class="c-sub">Nations · Moderate Aid</div>
        <div class="c-metrics">
          <div class="c-metric"><span class="c-metric-label">Child Mortality</span><span class="c-metric-val">{dv['child_mort']:.1f}/1K</span></div>
          <div class="c-metric"><span class="c-metric-label">Avg Income</span><span class="c-metric-val">${dv['income']:,.0f}</span></div>
          <div class="c-metric"><span class="c-metric-label">GDP per Capita</span><span class="c-metric-val">${dv['gdpp']:,.0f}</span></div>
          <div class="c-metric"><span class="c-metric-label">Life Expectancy</span><span class="c-metric-val">{dv['life_expec']:.1f} yrs</span></div>
          <div class="c-metric"><span class="c-metric-label">Fertility Rate</span><span class="c-metric-val">{dv['total_fer']:.2f}</span></div>
        </div>
      </div>
      <div class="cluster-card c-developed">
        <span class="c-badge">Cluster · Low Priority</span>
        <div class="c-name">Developed</div>
        <div class="c-count">{int(dd['count'])}</div>
        <div class="c-sub">Nations · Self-Sustaining</div>
        <div class="c-metrics">
          <div class="c-metric"><span class="c-metric-label">Child Mortality</span><span class="c-metric-val">{dd['child_mort']:.1f}/1K</span></div>
          <div class="c-metric"><span class="c-metric-label">Avg Income</span><span class="c-metric-val">${dd['income']:,.0f}</span></div>
          <div class="c-metric"><span class="c-metric-label">GDP per Capita</span><span class="c-metric-val">${dd['gdpp']:,.0f}</span></div>
          <div class="c-metric"><span class="c-metric-label">Life Expectancy</span><span class="c-metric-val">{dd['life_expec']:.1f} yrs</span></div>
          <div class="c-metric"><span class="c-metric-label">Fertility Rate</span><span class="c-metric-val">{dd['total_fer']:.2f}</span></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Radar
    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">RADAR</span><h2>Multi-Dimensional Cluster Comparison</h2></div>', unsafe_allow_html=True)
    rf = ['child_mort','income','gdpp','life_expec','total_fer','health','exports']
    rl = ['Child Mortality','Income','GDP/Capita','Life Expectancy','Fertility','Health Spend','Exports']
    rdf = df.groupby('ClusterRole')[rf].mean()
    for c in rf:
        mn,mx = rdf[c].min(),rdf[c].max(); rdf[c] = (rdf[c]-mn)/(mx-mn+1e-9)
    fig_r = go.Figure()
    rc = {'Underdeveloped':('#C1502E','rgba(193,80,46,0.12)'),'Developing':('#D9A02C','rgba(217,160,44,0.12)'),'Developed':('#3E7A5E','rgba(62,122,94,0.12)')}
    for role in ['Underdeveloped','Developing','Developed']:
        if role in rdf.index:
            v = rdf.loc[role,rf].tolist(); v += [v[0]]
            lc,fc = rc[role]
            fig_r.add_trace(go.Scatterpolar(r=v,theta=rl+[rl[0]],name=role,
                line=dict(color=lc,width=2),fill='toself',fillcolor=fc))
    fig_r.update_layout(paper_bgcolor='#171D25',
        polar=dict(bgcolor='#10151B',
            radialaxis=dict(visible=True,range=[0,1],gridcolor='#38414C',color='#5B6470',tickfont=dict(size=8)),
            angularaxis=dict(gridcolor='#38414C',color='#5B6470')),
        legend=dict(bgcolor='#10151B',bordercolor='#38414C',borderwidth=1,font=dict(color='#C7C2B4')),
        font=dict(family='Inter',color='#5B6470'),height=440,margin=dict(l=60,r=60,t=20,b=20))
    st.plotly_chart(fig_r, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">DISTRIBUTION</span><h2>GDP per Capita</h2></div>', unsafe_allow_html=True)
        fg = px.box(df,x='ClusterRole',y='gdpp',color='ClusterRole',color_discrete_map=CMAP,
            labels={'gdpp':'GDP per Capita','ClusterRole':'Cluster'})
        fg.update_layout(**PL,height=340); st.plotly_chart(fg,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">DISTRIBUTION</span><h2>Life Expectancy</h2></div>', unsafe_allow_html=True)
        fl = px.box(df,x='ClusterRole',y='life_expec',color='ClusterRole',color_discrete_map=CMAP,
            labels={'life_expec':'Life Expectancy (yrs)','ClusterRole':'Cluster'})
        fl.update_layout(**PL,height=340); st.plotly_chart(fl,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 4 — PRIORITY INDEX
# ════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">RANKING</span><h2>Top 10 Countries Requiring Aid</h2></div>', unsafe_allow_html=True)
    c1,c2 = st.columns([2,3])
    with c1:
        mx = top10['Need_Score'].max()
        for i,(_, row) in enumerate(top10.iterrows()):
            pct = (row['Need_Score']/mx)*100
            st.markdown(f'<div class="prow"><div class="prow-rank">#{i+1:02d}</div><div class="prow-country">{row["country"]}</div><div class="prow-track"><div class="prow-fill" style="width:{pct:.0f}%"></div></div><div class="prow-score">{row["Need_Score"]:.1f}</div></div>', unsafe_allow_html=True)
    with c2:
        fb = px.bar(top10.sort_values('Need_Score'),x='Need_Score',y='country',orientation='h',
            color='Need_Score',color_continuous_scale=[[0,'#D9A02C'],[1,'#C1502E']],
            labels={'Need_Score':'Need Score','country':'Country'},text='Need_Score')
        fb.update_traces(texttemplate='%{text:.1f}',textposition='outside',textfont=dict(color='#F6F3EC',size=10),marker_line_width=0)
        fb.update_layout(**PL,height=420,coloraxis_showscale=False)
        st.plotly_chart(fb,use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Full table
    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">FULL TABLE</span><h2>All Countries — Searchable</h2></div>', unsafe_allow_html=True)
    srch = st.text_input("Search country","",placeholder="e.g. Haiti, India…")
    fclust = st.selectbox("Filter cluster",["All","Underdeveloped","Developing","Developed"])
    disp = df[['country','ClusterRole','child_mort','income','gdpp','life_expec','total_fer','Need_Score']].copy()
    disp.columns = ['Country','Cluster','Child Mortality','Income','GDP/Capita','Life Expectancy','Fertility','Need Score']
    disp = disp.sort_values('Need Score',ascending=False).reset_index(drop=True); disp.index+=1
    if srch: disp = disp[disp['Country'].str.contains(srch,case=False,na=False)]
    if fclust!="All": disp = disp[disp['Cluster']==fclust]
    st.dataframe(disp.style.background_gradient(subset=['Need Score'],cmap='Reds')
                 .format({'Child Mortality':'{:.1f}','Income':'${:,.0f}','GDP/Capita':'${:,.0f}','Life Expectancy':'{:.1f}','Fertility':'{:.2f}','Need Score':'{:.2f}'}),
                 use_container_width=True,height=400)
    buf=io.StringIO(); disp.to_csv(buf,index=False)
    st.download_button("⟲ Export CSV",buf.getvalue(),"priority_index.csv","text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

    # Bubble
    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">MACRO</span><h2>Income vs Child Mortality</h2></div>', unsafe_allow_html=True)
    fbb = px.scatter(df,x='income',y='child_mort',size='gdpp',color='ClusterRole',hover_name='country',
        color_discrete_map=CMAP,size_max=36,opacity=0.8,
        labels={'income':'Income (USD)','child_mort':'Child Mortality','ClusterRole':'Cluster'})
    fbb.update_layout(**PL,height=420)
    st.plotly_chart(fbb,use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 5 — AI INSIGHTS
# ════════════════════════════════════════════════════════
with tabs[4]:
    ud=srow('Underdeveloped'); dd=srow('Developed')
    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">EXECUTIVE BRIEFING</span><h2>AI-Derived Intelligence</h2></div>', unsafe_allow_html=True)
    for icon,tag,text in [
        ("●","Critical Finding",f"Cluster analysis identifies {int(ud['count'])} nations in the Underdeveloped tier with average child mortality of {ud['child_mort']:.1f} per 1,000 live births and average GDP per capita of just ${ud['gdpp']:,.0f}. These nations require immediate and sustained humanitarian intervention."),
        ("●","Priority Allocation","Haiti, Sierra Leone, and Chad emerge as the highest-need nations by composite Need Score, exhibiting simultaneously extreme child mortality, minimal income levels, and elevated fertility rates. Priority funding must be directed toward these nations first."),
        ("●","Healthcare Imperative",f"Nations in the Underdeveloped cluster record an average life expectancy of only {ud['life_expec']:.1f} years — a {dd['life_expec']-ud['life_expec']:.1f}-year gap versus Developed nations. Investment in healthcare infrastructure and maternal care is projected to yield the greatest humanitarian ROI."),
        ("●","PCA Interpretation",f"PC1 (Economic Development Dimension, {v1*100:.1f}% variance) shows strong separation between Underdeveloped and Developed clusters. PC2 (Trade Dimension) highlights import-export imbalances as a secondary differentiator."),
        ("●","Development Pathway","The Developing cluster represents nations in a transitional phase. Targeted investment in education, trade diversification, and healthcare can accelerate movement toward the Developed cluster within a 5–10 year horizon."),
        ("●","Strategic Recommendation","A tiered aid framework is recommended: (1) Emergency humanitarian relief for the top Critical nations; (2) Long-term development finance for Developing nations; (3) Trade and institutional capacity building for Developed nations facing vulnerability."),
    ]:
        st.markdown(f'<div class="insight-card"><div class="insight-tag"><span style="color:#D9A02C">{icon}</span>{tag}</div><div class="insight-text">{text}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 6 — FUND SIMULATOR
# ════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">SIMULATOR</span><h2>Humanitarian Fund Allocation</h2></div>', unsafe_allow_html=True)
    c1,c2 = st.columns([1,2])
    with c1:
        bopt = st.selectbox("Budget",["$10 Million","$20 Million","$50 Million","$100 Million","Custom"])
        bm = float(bopt.replace("$","").replace(" Million","")) if bopt!="Custom" else st.number_input("USD millions",1.0,1000.0,25.0,1.0)
        budget = bm*1_000_000
        nc = st.slider("Recipient countries",5,20,10)
        st.markdown(f'<div class="panel" style="margin-top:12px;text-align:center"><div class="kpi-label" style="justify-content:center">Total Budget</div><div class="kpi-val" style="color:#D9A02C">${bm:.0f}M</div><div class="kpi-sub">Across {nc} nations</div></div>', unsafe_allow_html=True)
    topn = df[df['ClusterRole']=='Underdeveloped'].sort_values('Need_Score',ascending=False).head(nc).copy()
    topn['Weight'] = topn['Need_Score']/topn['Need_Score'].sum()
    topn['Alloc'] = topn['Weight']*budget
    topn['AllocM'] = topn['Alloc']/1_000_000
    with c2:
        st.markdown('<div style="background:#171D25;border:1px solid #38414C;border-radius:4px;padding:16px">', unsafe_allow_html=True)
        mx = topn['Alloc'].max()
        for _,row in topn.iterrows():
            pct=(row['Alloc']/mx)*100
            st.markdown(f'<div class="arow"><div class="arow-country">{row["country"]}</div><div class="arow-track"><div class="arow-fill" style="width:{pct:.0f}%"></div></div><div class="arow-pct">{row["Weight"]*100:.1f}%</div><div class="arow-amt">${row["AllocM"]:.2f}M</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        fa = px.bar(topn.sort_values('Alloc'),x='AllocM',y='country',orientation='h',
            color='AllocM',color_continuous_scale=[[0,'#D9A02C'],[1,'#C1502E']],
            labels={'AllocM':'Allocation ($M)','country':'Country'})
        fa.update_traces(marker_line_width=0)
        fa.update_layout(**PL,height=380,coloraxis_showscale=False,
            title=dict(text='Funding Allocation',font=dict(color='#F6F3EC',size=13)))
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(fa,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        fp = go.Figure(go.Pie(labels=topn['country'],values=topn['Alloc'],hole=0.55,
            marker=dict(colors=px.colors.sequential.Oranges_r[:len(topn)],line=dict(color='#10151B',width=1.5)),
            textfont=dict(color='#F6F3EC',size=9)))
        fp.update_layout(**PL,height=380,
            title=dict(text='Share Distribution',font=dict(color='#F6F3EC',size=13)),
            annotations=[dict(text=f'${bm:.0f}M',x=0.5,y=0.5,font_size=16,font_color='#F6F3EC',showarrow=False)])
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(fp,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    ae = topn[['country','Need_Score','Weight','Alloc']].copy()
    ae.columns=['Country','Need Score','Weight','Allocation (USD)']
    ae['Weight']=(ae['Weight']*100).round(2); ae['Allocation (USD)']=ae['Allocation (USD)'].round(2)
    buf2=io.StringIO(); ae.to_csv(buf2,index=False)
    st.download_button("⟲ Export Allocation Plan",buf2.getvalue(),"allocation.csv","text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 7 — ANALYTICS SUITE
# ════════════════════════════════════════════════════════
with tabs[6]:
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">HISTOGRAM</span><h2>Child Mortality</h2></div>', unsafe_allow_html=True)
        fh2 = px.histogram(df,x='child_mort',color='ClusterRole',nbins=28,
            color_discrete_map=CMAP,barmode='overlay',opacity=0.72,
            labels={'child_mort':'Child Mortality','ClusterRole':'Cluster'})
        fh2.update_layout(**PL,height=320); st.plotly_chart(fh2,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">VIOLIN</span><h2>GDP Distribution</h2></div>', unsafe_allow_html=True)
        fv = px.violin(df,x='ClusterRole',y='gdpp',color='ClusterRole',box=True,points='outliers',
            color_discrete_map=CMAP,labels={'gdpp':'GDP per Capita','ClusterRole':'Cluster'})
        fv.update_layout(**PL,height=320); st.plotly_chart(fv,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">SCATTER</span><h2>GDP vs Life Expectancy</h2></div>', unsafe_allow_html=True)
        fs = px.scatter(df,x='gdpp',y='life_expec',size='child_mort',color='ClusterRole',hover_name='country',
            color_discrete_map=CMAP,size_max=22,opacity=0.8,
            labels={'gdpp':'GDP/Capita','life_expec':'Life Expectancy','ClusterRole':'Cluster'})
        fs.update_layout(**PL,height=360); st.plotly_chart(fs,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">SCATTER</span><h2>Fertility vs Child Mortality</h2></div>', unsafe_allow_html=True)
        ff = px.scatter(df,x='total_fer',y='child_mort',color='ClusterRole',hover_name='country',
            color_discrete_map=CMAP,opacity=0.85,
            labels={'total_fer':'Fertility Rate','child_mort':'Child Mortality','ClusterRole':'Cluster'})
        ff.update_traces(marker=dict(size=7))
        ff.update_layout(**PL,height=360); st.plotly_chart(ff,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-head"><span class="panel-num">EXPLORER</span><h2>Any Indicator by Cluster</h2></div>', unsafe_allow_html=True)
    fsel = st.selectbox("Select indicator",feat_cols)
    fe = px.box(df,x='ClusterRole',y=fsel,color='ClusterRole',points='all',hover_name='country',
        color_discrete_map=CMAP,labels={fsel:fsel.replace('_',' ').title(),'ClusterRole':'Cluster'})
    fe.update_layout(**PL,height=400); st.plotly_chart(fe,use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #38414C;margin-top:3rem;padding-top:16px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <span style="font-family:'Source Serif 4',serif;font-size:0.85rem;color:#5B6470;">HELP <em>International</em></span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;color:#38414C;text-transform:uppercase;">HNDCSAI26.1 · PCA + K-Means · 167 Countries · 9 Indicators · 3 Clusters</span>
</div>
""", unsafe_allow_html=True)
