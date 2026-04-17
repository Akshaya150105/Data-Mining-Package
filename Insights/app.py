import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Aadhaar Data Mining",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 15px;
}

/* ── App background ── */
[data-testid="stAppViewContainer"] {
    background: #f4f3ef;
}
[data-testid="stAppViewContainer"] > .main > div {
    padding-top: 1.8rem;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0c1220 !important;
    border-right: 1px solid #1e2840;
}
[data-testid="stSidebar"] * {
    color: #a8b3cc !important;
}
[data-testid="stSidebar"] .stRadio > label {
    color: #a8b3cc !important;
    font-size: 13px;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #a8b3cc !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #cbd5e8 !important;
    font-size: 13.5px !important;
    padding: 6px 10px;
    border-radius: 8px;
    margin-bottom: 2px;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-testid="stMarkdownContainer"] {
    background: rgba(99, 140, 255, 0.12) !important;
    color: #638cff !important;
}

/* ── Section header ── */
.page-header {
    margin-bottom: 0.3rem;
}
.page-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 6px;
}
.page-sub {
    font-size: 13.5px;
    color: #64748b;
    font-weight: 300;
    margin-bottom: 1.6rem;
    line-height: 1.6;
    border-left: 3px solid #e2e0d8;
    padding-left: 10px;
}

/* ── Metric strip ── */
.metric-strip {
    display: flex;
    gap: 12px;
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 130px;
    background: #ffffff;
    border: 1px solid #e8e6e0;
    border-radius: 14px;
    padding: 16px 18px 14px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #638cff, #a78bfa);
    border-radius: 14px 14px 0 0;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.metric-label {
    font-size: 10.5px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 7px;
}
.metric-value {
    font-family: 'Sora', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1;
    margin-bottom: 5px;
}
.metric-value-sm {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 5px;
}
.metric-note {
    font-size: 11.5px;
    color: #94a3b8;
    font-weight: 300;
}

/* ── Insight cards ── */
.icard {
    background: #ffffff;
    border: 1px solid #e8e6e0;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left-width: 3px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: transform 0.15s, box-shadow 0.15s;
}
.icard:hover {
    transform: translateX(2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.07);
}
.icard-cluster {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 5px;
}
.icard-title {
    font-family: 'Sora', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 5px;
}
.icard-body {
    font-size: 12.5px;
    color: #64748b;
    line-height: 1.65;
    font-weight: 300;
}

/* ── Image frame ── */
.img-frame {
    background: #ffffff;
    border: 1px solid #e8e6e0;
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    overflow: hidden;
}

/* ── Note / callout box ── */
.note-box {
    background: linear-gradient(135deg, #f8faff 0%, #f3f4f6 100%);
    border: 1px solid #e2e8f0;
    border-left: 3px solid #638cff;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 13px;
    color: #475569;
    line-height: 1.75;
    margin-top: 10px;
}
.note-box b {
    color: #1e293b;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #e8e6e0;
    margin: 0.8rem 0 1.2rem;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 2px solid #e8e6e0;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Sora', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: #94a3b8;
    padding: 8px 16px;
    border-radius: 8px 8px 0 0;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: color 0.15s, background 0.15s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #1e293b;
    background: rgba(0,0,0,0.03);
}
.stTabs [aria-selected="true"] {
    color: #638cff !important;
    background: rgba(99, 140, 255, 0.06) !important;
    border-bottom: 2px solid #638cff !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1rem;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e8e6e0;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'Sora', sans-serif;
    font-size: 13.5px;
    font-weight: 500;
    color: #334155;
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e8e6e0;
}

/* ── Code / metric accent ── */
.mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    background: #f1f5f9;
    padding: 2px 6px;
    border-radius: 4px;
    color: #3b82f6;
}

/* ── Legend item (sidebar style) ── */
.legend-label {
    font-size: 13px;
    font-weight: 500;
    color: #1e293b;
    margin-bottom: 0.6rem;
}

/* ── Interp card system ── */
.interp-section { margin-top: 1.6rem; }
.interp-header {
    font-size: 12px; font-weight: 700; color: #475569;
    text-transform: uppercase; letter-spacing: .08em;
    margin-bottom: 1rem; padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
    font-family: 'Sora', sans-serif;
}
.interp-card {
    background: white; border: 1px solid #e8e6e0;
    border-radius: 12px; padding: 14px 18px; margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.interp-card-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
}
.comp-badge {
    font-size: 10px; font-weight: 700; color: white;
    background: #638cff; border-radius: 6px;
    padding: 3px 10px; letter-spacing: .05em;
    white-space: nowrap; font-family: 'Sora', sans-serif;
}
.comp-title {
    font-size: 13.5px; font-weight: 600; color: #1e293b;
    font-family: 'Sora', sans-serif;
}
.comp-body { font-size: 12.5px; color: #64748b; line-height: 1.7; font-weight: 300; }
.comp-meta { display: flex; gap: 16px; margin-top: 8px; flex-wrap: wrap; }
.comp-meta-item { font-size: 12px; color: #94a3b8; }
.comp-meta-item b { color: #475569; }
.interp-note {
    background: #f0f4ff; border-left: 3px solid #638cff;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: 12.5px; color: #374151; line-height: 1.7;
    margin-top: 1rem;
}

/* ── Selectbox + text input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    border-radius: 8px !important;
    border-color: #e2e8f0 !important;
    font-size: 13.5px !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #0f172a !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
    font-family: 'Sora', sans-serif !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #1e293b !important;
}

/* ── Warning / info ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 13px !important;
}

/* ── Metrics widget ── */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e8e6e0;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Directory paths ─────────────────────────────────────────────────────────
CLUSTER_DIR    = Path("clustering_output")
STATE_DIR      = Path("state_output")
TIMESERIES_DIR = Path("timeseries_output")
SPATIAL_DIR    = Path("spatial_output")
TABLE_DIR      = Path("table_output")
STGCN_BASE_DIR = Path("..") / "STGCN"
BIO_MODEL_DIR  = Path("../STGCN/biometric_model_output")
ENROL_MODEL_DIR= Path("../STGCN/enrolment_model_output")
TENSOR_DIR     = Path("tensor_output")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='padding: 10px 0 24px;'>
            <div style='font-family: Sora, sans-serif; font-size: 1.2rem;
                        font-weight: 700; color: #e2e8f0; line-height: 1.3;
                        letter-spacing: -0.01em; margin-bottom: 4px;'>
                Aadhaar<br>Data Mining
            </div>
            <div style='font-size: 10px; color: #4a5568; font-weight: 600;
                        letter-spacing: 0.1em; text-transform: uppercase;'>
                Research Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<hr style='border:none;border-top:1px solid #1e2840;margin:0 0 16px;'>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Clustering Analysis",
            "Table Analysis",
            "State Comparison",
            "Time Series Trends",
            "Spatial Autocorrelation",
            "District Deep-Dive",
            "STGCN Results",
            "Tensor Decomposition",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size: 11.5px; color: #3d4f6b; padding: 14px 16px;
                    background: #111827; border-radius: 10px; line-height: 1.8;
                    border: 1px solid #1e2840;'>
            <div style='font-weight:600; color:#638cff; margin-bottom:6px;
                        font-size:10px; text-transform:uppercase; letter-spacing:.08em;'>
                Data Source
            </div>
            UIDAI Aadhaar open data<br>
            Apr – Jul 2025<br>
            775 districts · 3 tables<br>
            ~5 M records
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Helper components ────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class='page-header'>
            <div class='page-title'>{title}</div>
            <div class='page-sub'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def img(path):
    if Path(path).exists():
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(str(path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ `{Path(path).name}` not found — run the analysis script first.")


def note(title_or_body: str, body: str = ""):
    """Pass (title, body) or just (body,)."""
    if body:
        content = f"<b>{title_or_body}</b><br><br>{body}"
    else:
        content = title_or_body
    st.markdown(f"<div class='note-box'>{content}</div>", unsafe_allow_html=True)


def icard(color: str, cluster_label: str, title: str, body: str):
    st.markdown(
        f"""
        <div class='icard' style='border-left-color:{color};'>
            <div class='icard-cluster' style='color:{color};'>{cluster_label}</div>
            <div class='icard-title'>{title}</div>
            <div class='icard-body'>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note_text="", small=False):
    val_class = "metric-value-sm" if small else "metric-value"
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='{val_class}'>{value}</div>"
        f"<div class='metric-note'>{note_text}</div>"
        f"</div>"
    )


def metric_strip(*cards_html):
    inner = "".join(cards_html)
    st.markdown(f"<div class='metric-strip'>{inner}</div>", unsafe_allow_html=True)


# ── INTERP CSS (shared across Tensor pages) ──────────────────────────────────
INTERP_CSS = """<style>
.interp-section{margin-top:1.6rem}
.interp-header{font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;
               letter-spacing:.08em;margin-bottom:1rem;padding-bottom:8px;
               border-bottom:2px solid #e2e8f0;font-family:'Sora',sans-serif}
.interp-card{background:white;border:1px solid #e8e6e0;border-radius:12px;
             padding:14px 18px;margin-bottom:8px;box-shadow:0 1px 3px rgba(0,0,0,.03)}
.interp-card-header{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.comp-badge{font-size:10px;font-weight:700;color:white;background:#638cff;border-radius:6px;
            padding:3px 10px;letter-spacing:.05em;white-space:nowrap;font-family:'Sora',sans-serif}
.comp-title{font-size:13.5px;font-weight:600;color:#1e293b;font-family:'Sora',sans-serif}
.comp-body{font-size:12.5px;color:#64748b;line-height:1.7;font-weight:300}
.comp-meta{display:flex;gap:16px;margin-top:8px;flex-wrap:wrap}
.comp-meta-item{font-size:12px;color:#94a3b8}
.comp-meta-item b{color:#475569}
.interp-note{background:#f0f4ff;border-left:3px solid #638cff;border-radius:0 8px 8px 0;
             padding:12px 16px;font-size:12.5px;color:#374151;line-height:1.7;margin-top:1rem}
</style>"""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — CLUSTERING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if page == "Clustering Analysis":
    page_header(
        "District Clustering Analysis",
        "K-Means and DBSCAN applied to engineered Aadhaar adoption features across 775 Indian districts · Apr–Jul 2025",
    )

    metric_strip(
        metric_card("Districts", "775", "28 states + UTs"),
        metric_card("Features used", "18", "ratios, ranks, growth"),
        metric_card("K-Means K", "5", "silhouette = 0.303"),
        metric_card("DBSCAN noise", "40", "outlier districts"),
        metric_card("Time steps", "4", "monthly snapshots"),
    )

    t1, t2, t3, t4, t5 = st.tabs(
        ["K-Means Map", "DBSCAN Map", "Cluster Profiles", "Elbow / Silhouette", "PCA Scatter"]
    )

    with t1:
        c1, c2 = st.columns([2, 1])
        with c1:
            img(CLUSTER_DIR / "kmeans_choropleth.png")
            h = CLUSTER_DIR / "kmeans_choropleth.html"
            if h.exists():
                st.info(f"🗺️ Interactive version → open `{h}` in your browser to hover over districts.")
        with c2:
            st.markdown("<div class='legend-label'>Cluster legend</div>", unsafe_allow_html=True)
            for n, col, title, desc in [
                ("0", "#22c55e", "Mainstream Coverage",
                 "Balanced ratios, stable trends. South India, Rajasthan periphery, Northeast."),
                ("1", "#638cff", "High-Dependency Zones",
                 "High child-to-adult ratio. UP, Bihar, MP — younger demographic profile."),
                ("2", "#f97316", "High Growth Momentum",
                 "Elevated daily % change. Rapid enrolment spikes from local campaigns."),
                ("3", "#eab308", "Adult-Dominant Pattern",
                 "High adult ratio. South India, Gujarat — mature Aadhaar saturation."),
                ("4", "#ec4899", "Restricted / Border Zones",
                 "J&K, Ladakh. Low ratios, high volatility — different administration."),
            ]:
                icard(col, f"Cluster {n}", title, desc)

    with t2:
        c1, c2 = st.columns([2, 1])
        with c1:
            img(CLUSTER_DIR / "dbscan_choropleth.png")
        with c2:
            st.markdown("<div class='legend-label'>What DBSCAN reveals</div>", unsafe_allow_html=True)
            icard("#638cff", "Cluster 1 · 726 districts", "The Mainstream",
                  "94% of districts share one dense cluster — uniform Aadhaar adoption across India.")
            icard("#22c55e", "Cluster 0 · 9 districts", "Metro Outlier Cluster",
                  "9 districts distinct from mainstream — likely high-volume urban centres.")
            icard("#94a3b8", "Noise · 40 districts", "True Outliers",
                  "40 districts fit no cluster. Check cluster_summary.csv where dbscan_cluster == −1.")

    with t3:
        img(CLUSTER_DIR / "cluster_profiles_kmeans.png")
        csv = CLUSTER_DIR / "cluster_summary.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            counts = (
                df.groupby("kmeans_cluster")["district"]
                .count()
                .reset_index()
                .rename(columns={"district": "Districts", "kmeans_cluster": "Cluster"})
            )
            counts["Cluster"] = counts["Cluster"].apply(lambda x: f"Cluster {x}")
            st.dataframe(counts, hide_index=True, use_container_width=False)
            with st.expander("Browse all districts"):
                search = st.text_input("Search district or state", "")
                cols = [c for c in ["district", "state", "kmeans_cluster", "dbscan_cluster"] if c in df.columns]
                filtered = df[cols]
                if search:
                    mask = filtered["district"].str.contains(search, case=False, na=False) | \
                           filtered["state"].str.contains(search, case=False, na=False)
                    filtered = filtered[mask]
                st.dataframe(filtered, hide_index=True, use_container_width=True)

    with t4:
        img(CLUSTER_DIR / "elbow_silhouette.png")
        note(
            "Choosing K",
            "The <b>elbow curve</b> shows inertia dropping as K increases — the bend marks diminishing returns. "
            "The <b>silhouette score</b> measures cluster separation (1.0 = perfect). We used <b>K = 5</b> for "
            "a balance of statistical optimality and geographic interpretability.",
        )

    with t5:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**K-Means — PCA space**")
            img(CLUSTER_DIR / "pca_kmeans.png")
        with c2:
            st.markdown("**DBSCAN — PCA space**")
            img(CLUSTER_DIR / "pca_dbscan.png")
        note(
            "Reading the PCA Scatter",
            "Each dot is one district projected from 18 features to 2D. PCA captures 42.8% of total variance — "
            "overlap in this view doesn't mean bad clusters; separation exists in higher dimensions.",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — TABLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Table Analysis":
    page_header(
        "Table Analysis",
        "Complete statistics and visualizations for all 3 Aadhaar tables across all states and all dates",
    )

    TD = TABLE_DIR
    if not (TD / "bio_growth_trend.png").exists():
        st.warning("Charts not found. Run `table_analysis.py` first.")
        st.code("python table_analysis.py --db ../database/aadhar.duckdb")
        st.stop()

    def show(path, wide=True):
        p = TD / path
        if p.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(p), use_container_width=wide)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ `{path}` not found.")

    def tbl_note(body):
        st.markdown(
            f"<div class='note-box'>{body}</div>",
            unsafe_allow_html=True,
        )

    tab_bio, tab_enrol, tab_demo, tab_combined = st.tabs(
        ["Biometric", "Enrolment", "Demographic", "Combined"]
    )

    with tab_bio:
        st.markdown(
            "<div style='font-size:13px;color:#64748b;margin-bottom:14px;'>"
            "Biometric enrolment across all states and all dates — age structure, "
            "dependency ratios, growth momentum and volatility.</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**National age group totals**")
            show("bio_age_distribution.png")
        with c2:
            st.markdown("**Top 10 vs Bottom 10 states**")
            show("bio_top_bottom.png")
        st.markdown("**National daily trend — bio total with age breakdown**")
        show("bio_growth_trend.png")
        tbl_note("Bars show raw daily totals. Lines show 7-day rolling averages for total, age 5–17 and age 17+. Vertical lines mark month boundaries.")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**State × month dependency heatmap**")
            show("bio_state_heatmap.png")
            tbl_note("Each cell = avg biometric dependency ratio. Darker = higher ratio of young enrolees.")
        with c4:
            st.markdown("**Enrolment volatility by state**")
            show("bio_volatility_map.png")
            tbl_note("7-day std of biometric enrolment per state. Orange = above-median volatility.")
        st.markdown("**Age 5–17 ratio vs dependency — state scatter**")
        show("bio_ratio_scatter.png")
        tbl_note("Each dot is one state. Bubble size = total bio enrolment. Colour = avg daily growth rate.")

    with tab_enrol:
        st.markdown(
            "<div style='font-size:13px;color:#64748b;margin-bottom:14px;'>"
            "Enrolment across all states and all dates — age breakdowns, adult vs minor ratios, growth and volatility patterns.</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**National age group share**")
            show("enrol_age_pie.png")
        with c2:
            st.markdown("**Adult vs minor ratio — all states**")
            show("enrol_adult_minor_bar.png")
            tbl_note("Sorted by adult ratio. States at the top have higher adult saturation.")
        st.markdown("**National daily trend — enrolment total by age group**")
        show("enrol_trend.png")
        tbl_note("All 4 series smoothed with a 7-day rolling average.")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**State × month growth rate heatmap**")
            show("enrol_growth_heatmap.png")
            tbl_note("Green = positive growth, Red = decline. Cells = avg daily % change.")
        with c4:
            st.markdown("**Growth momentum ranking**")
            show("enrol_top_growth.png")
            tbl_note("Orange = above-median growth. Purple dashed line = national median.")
        st.markdown("**Enrolment volatility by state**")
        show("enrol_volatility.png")
        tbl_note("High volatility = irregular enrolment bursts rather than steady flow.")

    with tab_demo:
        st.markdown(
            "<div style='font-size:13px;color:#64748b;margin-bottom:14px;'>"
            "Demographic enrolment across all states and all dates — age ratios and dependency patterns.</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dependency ratio — distribution + state ranking**")
            show("demo_dependency_dist.png")
            tbl_note("Left: distribution across states. Right: per-state ranking. Orange = above national median.")
        with c2:
            st.markdown("**Age group split — top 30 states**")
            show("demo_state_comparison.png")
            tbl_note("Stacked bar showing age 5–17 vs age 17+ demographic enrolment.")
        st.markdown("**National daily trend — demographic total by age group**")
        show("demo_trend.png")
        tbl_note("Smoothed daily totals for demographic enrolment.")
        st.markdown("**State × month age 5–17 ratio heatmap**")
        show("demo_age_ratio_heatmap.png")
        tbl_note("Darker cells = higher proportion of age 5–17. Consistent dark rows = structurally young districts.")

    with tab_combined:
        st.markdown(
            "<div style='font-size:13px;color:#64748b;margin-bottom:14px;'>"
            "Cross-table analysis — all 3 tables on the same timeline and a feature correlation matrix.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**All 3 tables — national daily totals on the same chart**")
        show("all_tables_trend.png")
        tbl_note("Biometric (purple) · Enrolment (green) · Demographic (amber). All smoothed with a 7-day rolling average.")
        st.markdown("**Feature correlation matrix — Bio × Enrolment × Demographic**")
        show("correlation_heatmap.png")
        tbl_note(
            "Lower triangle only. <b>Green (r→1)</b> = strong positive correlation. "
            "<b>Red (r→−1)</b> = strong negative. <b>Near-zero</b> = independent features."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — STATE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "State Comparison":
    page_header(
        "State-level Comparison",
        "Rankings, age ratios, growth momentum and dependency patterns across all Indian states · Apr–Jul 2025",
    )

    csv_path = STATE_DIR / "state_summary.csv"
    if not csv_path.exists():
        st.warning("Run `state_comparison.py` first to generate charts and data.")
        st.code("python state_comparison.py --db ../database/aadhar.duckdb")
        st.stop()

    sdf = pd.read_csv(csv_path)
    top_enrol  = sdf.loc[sdf["enrol_total"].idxmax(), "state"]
    top_growth = sdf.loc[sdf["avg_enrol_growth"].idxmax(), "state"]
    top_dep    = sdf.loc[sdf["avg_dependency_ratio"].idxmax(), "state"]
    top_adult  = sdf.loc[sdf["avg_adult_ratio"].idxmax(), "state"]

    metric_strip(
        metric_card("States analysed", str(len(sdf)), "+ union territories"),
        metric_card("Highest enrolment", top_enrol[:18], "total volume Apr–Jul", small=True),
        metric_card("Fastest growing", top_growth[:18], "highest avg daily growth %", small=True),
        metric_card("Highest dependency", top_dep[:18], "child-to-adult ratio", small=True),
        metric_card("Most adult-saturated", top_adult[:18], "highest adult enrol ratio", small=True),
    )

    def show_img(path):
        p = Path(path)
        if p.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(p), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ `{p.name}` not found — run state_comparison.py first.")

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Enrolment Ranking", "Adult vs Minor", "Growth Momentum", "Dependency Heatmap", "Size vs Growth", "Data Table"]
    )

    with t1:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(STATE_DIR / "state_enrolment_bar.png")
        with c2:
            note(
                "What this shows",
                "Total Aadhaar enrolment summed across all districts within each state over Apr–Jul 2025. "
                "<b style='color:#638cff;'>Highlighted bars</b> = top 5 states by volume. "
                "Use other tabs for size-agnostic comparisons.",
            )

    with t2:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(STATE_DIR / "state_adult_vs_minor.png")
        with c2:
            note(
                "Adult vs minor ratio",
                "Split between minor (0–17) and adult (18+) enrolments as a proportion of state total.<br><br>"
                "<b>High minor ratio</b> = younger population or ongoing child Aadhaar push.<br><br>"
                "<b>High adult ratio</b> = mature, near-saturated adult coverage.",
            )

    with t3:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(STATE_DIR / "state_growth_bar.png")
        with c2:
            note(
                "Growth Momentum",
                "Average daily % change in enrolment across all districts per state. "
                "<b style='color:#f97316;'>Orange</b> = above-median growth. "
                "<b style='color:#94a3b8;'>Grey</b> = below-median. "
                "Dashed line = national median.",
            )

    with t4:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(STATE_DIR / "state_dependency_heatmap.png")
        with c2:
            note(
                "Dependency Heatmap",
                "Child-to-adult biometric dependency ratio per state across 4 monthly time steps.<br><br>"
                "<b>Darker red</b> = higher proportion of children vs adults.<br>"
                "<b>Lighter yellow</b> = adult-dominant.<br><br>"
                "Consistent dark across months = structural demographic pattern.",
            )

    with t5:
        show_img(STATE_DIR / "state_scatter.png")
        note(
            "How to read this chart",
            "X = number of districts. Y = avg daily growth %. "
            "Bubble size = total enrolment. Colour = adult ratio.<br><br>"
            "<b>Top-left</b>: small states, high growth — campaigns working efficiently.<br>"
            "<b>Bottom-right</b>: large states, slower growth — scale challenge.",
        )

    with t6:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            search_state = st.text_input("Search state", "")
        with col_f2:
            sort_col = st.selectbox(
                "Sort by",
                ["enrol_total", "avg_enrol_growth", "avg_adult_ratio",
                 "avg_minor_ratio", "avg_dependency_ratio", "district_count"],
            )
        disp = sdf.copy()
        float_cols = disp.select_dtypes("float").columns
        disp[float_cols] = disp[float_cols].round(4)
        if search_state:
            disp = disp[disp["state"].str.contains(search_state, case=False, na=False)]
        disp = disp.sort_values(sort_col, ascending=False)
        st.dataframe(disp, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", data=disp.to_csv(index=False),
                           file_name="state_summary.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — TIME SERIES TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Time Series Trends":
    page_header(
        "Time Series Trends",
        "Aadhaar enrolment patterns over 70 dates · Apr–Dec 2025 · national daily totals, state trends, heatmap, growth and volatility",
    )

    if not (TIMESERIES_DIR / "ts_national.png").exists():
        st.warning("Time series charts not found. Run `time_series.py` first.")
        st.code("python time_series.py --db ../database/aadhar.duckdb")
        st.stop()

    metric_strip(
        metric_card("Date range", "Apr–Dec", "2025 · 9 months"),
        metric_card("Total time steps", "70", "common across 3 tables"),
        metric_card("Granularity", "Daily", "Sep–Dec · Monthly Apr–Jul"),
        metric_card("Metric tracked", "Enrol", "+ biometric overlay"),
        metric_card("Smoothing", "7-day", "rolling average applied"),
    )

    def show_img(path):
        p = Path(path)
        if p.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(p), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ `{p.name}` not found.")

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["National Trend", "Top 10 States", "State Heatmap", "Monthly Growth", "Volatility", "Raw Data"]
    )

    with t1:
        show_img(TIMESERIES_DIR / "ts_national.png")
        note(
            "National Enrolment Trend",
            "Total Aadhaar enrolment summed across all states per day. Light bars = raw daily values. "
            "Solid line = 7-day rolling average. Dashed line = biometric enrolment overlay.<br><br>"
            "Gap between Apr–Jul and Sep is expected — monthly snapshots vs daily records.",
        )

    with t2:
        show_img(TIMESERIES_DIR / "ts_top10_states.png")
        note(
            "Top 10 States by Total Enrolment",
            "Each line = one state's enrolment over time, smoothed with a 3-point rolling average. "
            "Crossing lines indicate states overtaking each other in pace.",
        )

    with t3:
        show_img(TIMESERIES_DIR / "ts_heatmap.png")
        note(
            "State × Date Intensity Heatmap",
            "Each row = a state, each column = a date. Colour intensity normalised per state. "
            "<b>Dark red columns</b> = unusually high enrolment day across many states. "
            "<b>Dark red rows</b> = consistently at peak throughout the period.",
        )

    with t4:
        show_img(TIMESERIES_DIR / "ts_monthly_growth.png")
        note(
            "Month-over-Month Growth Rate",
            "% change in enrolment from one month to the next for top 20 states. "
            "Bar above 0 = growth. Sep spike is partly a data-density effect from monthly → daily transition.",
        )

    with t5:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(TIMESERIES_DIR / "ts_volatility.png")
        with c2:
            note(
                "Enrolment Volatility",
                "Avg 7-day std in enrolment per state.<br><br>"
                "<b style='color:#f97316;'>Orange</b> = above-median volatility — campaign-driven spikes.<br><br>"
                "<b style='color:#94a3b8;'>Grey</b> = steady, predictable enrolment.",
            )

    with t6:
        csv_path = TIMESERIES_DIR / "ts_data.csv"
        if csv_path.exists():
            tsdf = pd.read_csv(csv_path, index_col=0)
            st.markdown(f"**State × date enrolment pivot ({len(tsdf)} states × {len(tsdf.columns)} dates)**")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                search_ts = st.text_input("Search state", "", key="ts_search")
            with col_f2:
                sort_ts = st.selectbox("Sort by total", ["Descending", "Ascending"])
            disp = tsdf.copy()
            if search_ts:
                disp = disp[disp.index.str.contains(search_ts, case=False, na=False)]
            disp["_total"] = disp.sum(axis=1)
            disp = disp.sort_values("_total", ascending=(sort_ts == "Ascending")).drop(columns=["_total"])
            st.dataframe(disp, use_container_width=True)
            st.download_button("Download ts_data.csv", data=disp.to_csv(),
                               file_name="ts_data.csv", mime="text/csv")
        else:
            st.warning("`ts_data.csv` not found in timeseries_output/")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — SPATIAL AUTOCORRELATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Spatial Autocorrelation":
    page_header(
        "Spatial Autocorrelation",
        "Moran's I analysis — does high Aadhaar enrolment cluster geographically or is it randomly distributed?",
    )

    if not (SPATIAL_DIR / "moran_scatter.png").exists():
        st.warning("Spatial analysis outputs not found. Run `spatial_autocorr.py` first.")
        st.stop()

    report_path = SPATIAL_DIR / "moran_report.txt"
    global_I, global_p = None, None
    hh_count = ll_count = hl_count = lh_count = 0
    if report_path.exists():
        txt = report_path.read_text()
        for line in txt.split("\n"):
            if "Global Moran" in line and ":" in line:
                try: global_I = float(line.split(":")[-1].strip())
                except: pass
            if "P-value" in line and ":" in line:
                try: global_p = float(line.split(":")[-1].strip())
                except: pass
            if "HH (hot spots)" in line:
                try: hh_count = int(line.split(":")[1].strip().split()[0])
                except: pass
            if "LL (cold spots)" in line:
                try: ll_count = int(line.split(":")[1].strip().split()[0])
                except: pass
            if "HL (outliers)" in line and "lh" not in line.lower():
                try: hl_count = int(line.split(":")[1].strip().split()[0])
                except: pass
            if "LH (outliers)" in line:
                try: lh_count = int(line.split(":")[1].strip().split()[0])
                except: pass

    interp = "Spatially clustered" if global_I and global_I > 0 else "Spatially dispersed"
    sig_label = ""
    if global_p is not None:
        sig_label = ("p < 0.001 ***" if global_p < 0.001 else
                     "p < 0.01 **" if global_p < 0.01 else
                     "p < 0.05 *" if global_p < 0.05 else "not significant")

    metric_strip(
        metric_card("Global Moran's I", f"{global_I:.4f}" if global_I else "—", interp),
        metric_card("Significance", sig_label or "—", "permutation test (999 runs)", small=True),
        metric_card("Hot spots (HH)", str(hh_count), "high surrounded by high"),
        metric_card("Cold spots (LL)", str(ll_count), "low surrounded by low"),
        metric_card("Spatial outliers", str(hl_count + lh_count), "HL + LH districts"),
    )

    def show_img(path):
        p = Path(path)
        if p.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(p), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ `{p.name}` not found.")

    t1, t2, t3, t4, t5 = st.tabs(
        ["LISA Map", "Moran Scatter", "All Features", "Report", "District Data"]
    )

    with t1:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(SPATIAL_DIR / "lisa_map.png")
        with c2:
            st.markdown("<div class='legend-label'>LISA cluster types</div>", unsafe_allow_html=True)
            for color, label, desc in [
                ("#ef4444", "HH — Hot spots",
                 "High enrolment surrounded by high neighbours. Geographically concentrated success zones."),
                ("#3b82f6", "LL — Cold spots",
                 "Low enrolment surrounded by low neighbours. Regions of consistently low adoption."),
                ("#f59e0b", "HL — Spatial outliers",
                 "High district surrounded by low neighbours. Isolated high performers."),
                ("#6ee7b7", "LH — Spatial outliers",
                 "Low district surrounded by high neighbours. Lagging behind its region — intervention candidates."),
                ("#e5e7eb", "NS — Not significant",
                 "No statistically significant spatial pattern."),
            ]:
                st.markdown(
                    f"<div class='icard' style='border-left-color:{color};'>"
                    f"<div class='icard-title'>{label}</div>"
                    f"<div class='icard-body'>{desc}</div></div>",
                    unsafe_allow_html=True,
                )

    with t2:
        c1, c2 = st.columns([3, 2])
        with c1:
            show_img(SPATIAL_DIR / "moran_scatter.png")
        with c2:
            note(
                "Reading the Moran Scatter",
                "Each dot = one district. X-axis = standardised enrolment. "
                "Y-axis = spatial lag (weighted avg of neighbours' enrolment).<br><br>"
                "<b>Slope = Global Moran's I.</b><br><br>"
                "Quadrants: Top-right = HH · Bottom-left = LL · Top-left = LH · Bottom-right = HL<br><br>"
                "Steep positive slope = strong geographic clustering.",
            )

    with t3:
        show_img(SPATIAL_DIR / "moran_by_feature.png")
        note(
            "Global Moran's I Across All Features",
            "<b style='color:#f97316;'>Orange (I > 0)</b> = feature clusters geographically.<br>"
            "<b style='color:#3b82f6;'>Blue (I < 0)</b> = feature is spatially dispersed.<br><br>"
            "Stars = statistical significance: * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001.",
        )

    with t4:
        if report_path.exists():
            st.markdown("**Full Moran's I report**")
            st.code(report_path.read_text(), language="text")
        else:
            st.warning("`moran_report.txt` not found.")

    with t5:
        csv_path = SPATIAL_DIR / "spatial_summary.csv"
        if csv_path.exists():
            sdf = pd.read_csv(csv_path)
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                search = st.text_input("Search district", "", key="spatial_search")
            with col_f2:
                lisa_filter = st.selectbox("Filter LISA type", ["All", "HH", "LL", "HL", "LH", "NS"])
            disp = sdf.copy()
            if search:
                disp = disp[disp["district"].str.contains(search, case=False, na=False)]
            if lisa_filter != "All":
                disp = disp[disp["lisa_type"] == lisa_filter]
            st.dataframe(disp.sort_values("enrol_total", ascending=False), hide_index=True, use_container_width=True)
            st.download_button("Download spatial_summary.csv", disp.to_csv(index=False),
                               "spatial_summary.csv", "text/csv")
        else:
            st.warning("`spatial_summary.csv` not found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — DISTRICT DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "District Deep-Dive":
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    import io

    page_header(
        "District Deep-Dive",
        "Select any district — complete KPI profile, 70-date time series, and state peer comparison",
    )

    prof_path = TABLE_DIR / "district_profiles.csv"
    ts_path   = TABLE_DIR / "district_timeseries.csv"

    if not prof_path.exists():
        st.warning("Run `table_analysis.py` first.")
        st.stop()

    @st.cache_data
    def load_dd():
        p = pd.read_csv(prof_path)
        t = pd.read_csv(ts_path, parse_dates=["date"])
        return p, t

    prof, ts = load_dd()
    BG = "#f4f3ef"

    cs, cd = st.columns([1, 2])
    with cs:
        sel_state = st.selectbox("State", sorted(prof["state"].dropna().unique()), key="dd_s")
    with cd:
        sel_dist = st.selectbox(
            "District",
            sorted(prof[prof["state"] == sel_state]["district"].dropna().unique()),
            key="dd_d",
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    row = prof[(prof["district"] == sel_dist) & (prof["state"] == sel_state)]
    if row.empty:
        st.warning("No data for this district.")
        st.stop()
    row = row.iloc[0]

    def val(col, fmt="num"):
        v = row.get(col, None)
        if v is None or (isinstance(v, float) and np.isnan(float(v))):
            return "—"
        v = float(v)
        if fmt == "num":  return f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.1f}K" if v >= 1e3 else f"{v:.0f}"
        if fmt == "pct":  return f"{v*100:.2f}%"
        if fmt == "dec2": return f"{v:.2f}"
        if fmt == "dec3": return f"{v:.3f}"
        if fmt == "rank": return f"#{int(v)}"
        return f"{v:.2f}"

    tier = row.get("performance_tier", "—")
    tierc = {
        "Top 10%": "#22c55e", "Top 25%": "#638cff", "Above average": "#eab308",
        "Below average": "#f97316", "Bottom 25%": "#94a3b8",
    }.get(tier, "#94a3b8")

    metric_strip(
        metric_card("Enrolment total", val("enrol_total"), "Apr–Jul 2025"),
        metric_card("Biometric total", val("bio_total"), "Apr–Jul 2025"),
        metric_card("Demographic total", val("demo_total"), "Apr–Jul 2025"),
        metric_card("National rank", val("enrol_total_national_rank", "rank"), f"of {len(prof)} districts"),
        metric_card("State rank", val("enrol_total_state_rank", "rank"), f"in {sel_state[:18]}"),
        metric_card(
            "Percentile",
            f"{val('enrol_total_percentile','dec2')}th",
            f"<span style='color:{tierc};font-weight:600;'>{tier}</span>",
        ),
    )

    c1, c2, c3 = st.columns(3)
    for col_obj, accent, cluster, body_items in [
        (c1, "#638cff", "Biometric", [
            ("Age 5–17", val("bio_age_5_17_total")),
            ("Age 17+", val("bio_age_17_plus_total")),
            ("Dependency", val("bio_dependency_ratio", "dec3")),
            ("7-day avg", val("bio_7day_avg")),
            ("Volatility", val("bio_7day_std", "dec2")),
            ("Daily growth", f"{val('bio_daily_pct_change','dec2')}%"),
            ("State rank", val("bio_rank_in_state", "dec2")),
        ]),
        (c2, "#22c55e", "Enrolment", [
            ("Age 0–5", val("enrol_age_0_5_total")),
            ("Age 5–17", val("enrol_age_5_17_total")),
            ("Age 18+", val("enrol_age_18_plus_total")),
            ("Minor ratio", val("enrol_minor_ratio", "pct")),
            ("Adult ratio", val("enrol_adult_ratio", "pct")),
            ("Daily growth", f"{val('enrol_daily_pct_change','dec2')}%"),
            ("Volatility", val("enrol_7day_std", "dec2")),
        ]),
        (c3, "#f59e0b", "Demographic", [
            ("Age 5–17", val("demo_age_5_17_total")),
            ("Age 17+", val("demo_age_17_plus_total")),
            ("Age 5 ratio", val("demo_age5_ratio", "pct")),
            ("Age 17 ratio", val("demo_age17_ratio", "pct")),
            ("Dependency", val("demo_dependency_ratio", "dec3")),
            ("Daily growth", f"{val('demo_daily_pct_change','dec2')}%"),
            ("State rank", val("demo_rank_in_state", "dec2")),
        ]),
    ]:
        body_html = "".join(f"<b>{k}:</b> {v}<br>" for k, v in body_items)
        col_obj.markdown(
            f"<div class='icard' style='border-left-color:{accent};'>"
            f"<div class='icard-cluster' style='color:{accent};'>{cluster}</div>"
            f"<div class='icard-body'>{body_html}</div></div>",
            unsafe_allow_html=True,
        )

    dist_ts = ts[(ts["district"] == sel_dist) & (ts["state"] == sel_state)].sort_values("date")
    if not dist_ts.empty:
        metric_sel = st.selectbox(
            "Feature to plot",
            ["enrol_total", "bio_total", "demo_total",
             "enrol_minor_ratio", "enrol_adult_ratio", "bio_dependency",
             "enrol_pct_change", "bio_pct_change"],
            key="dd_metric",
        )
        fig, axes = plt.subplots(
            2, 1, figsize=(13, 6), facecolor=BG,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )
        ax = axes[0]; ax.set_facecolor("#ffffff")
        vals   = dist_ts[metric_sel].fillna(0)
        smooth = vals.rolling(7, min_periods=1, center=True).mean()
        ax.bar(dist_ts["date"], vals, color="#638cff", alpha=0.15, width=0.8)
        ax.plot(dist_ts["date"], smooth, color="#638cff", linewidth=2.2)
        for ms in ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]:
            ax.axvline(pd.Timestamp(ms), color="#e8e6e0", linewidth=1)
            ax.text(pd.Timestamp(ms), smooth.max() * 1.01 if smooth.max() > 0 else 1,
                    f" {pd.Timestamp(ms).strftime('%b')}", fontsize=8, color="#94a3b8")
        ax.set_title(f"{sel_dist}  ·  {metric_sel}", fontsize=11, fontweight="bold", loc="left", pad=8,
                     fontfamily="sans-serif", color="#0f172a")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.3f}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        plt.setp(ax.get_xticklabels(), visible=False)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax.spines[sp].set_color("#e8e6e0")
        ax.grid(True, axis="y", color="#f1f5f9")

        ax2 = axes[1]; ax2.set_facecolor("#ffffff")
        g = dist_ts["enrol_pct_change"].fillna(0)
        ax2.bar(dist_ts["date"], g,
                color=["#22c55e" if x >= 0 else "#ef4444" for x in g],
                alpha=0.7, width=0.8)
        ax2.axhline(0, color="#94a3b8", linewidth=0.8)
        ax2.set_ylabel("Growth %", fontsize=8, color="#64748b")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        plt.setp(ax2.get_xticklabels(), rotation=28, ha="right", fontsize=7, color="#64748b")
        for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax2.spines[sp].set_color("#e8e6e0")
        ax2.grid(True, axis="y", color="#f1f5f9")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(); buf.seek(0)
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(buf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander(f"Compare with all districts in {sel_state}"):
        peers = prof[prof["state"] == sel_state].sort_values("enrol_total", ascending=False).copy()
        pcols = ["district", "enrol_total", "bio_total", "demo_total",
                 "enrol_minor_ratio", "enrol_adult_ratio", "bio_dependency_ratio",
                 "enrol_daily_pct_change", "enrol_total_state_rank"]
        pcols = [c for c in pcols if c in peers.columns]
        pdisp = peers[pcols].copy()
        pdisp[pdisp.select_dtypes("float").columns] = pdisp.select_dtypes("float").round(3)

        def hl(r):
            return ["background:#eef2ff;font-weight:500" if r["district"] == sel_dist else "" for _ in r]

        st.dataframe(pdisp.style.apply(hl, axis=1), hide_index=True, use_container_width=True)
        st.download_button("Download CSV", pdisp.to_csv(index=False),
                           f"{sel_state}_peers.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — STGCN RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "STGCN Results":
    import matplotlib.pyplot as plt
    import io

    page_header(
        "STGCN Results",
        "Weekly district-level forecasting results from biometric and enrolment STGCN models in ../STGCN",
    )

    MODEL_DIRS = {
        "Biometric Model": BIO_MODEL_DIR,
        "Enrolment Model": ENROL_MODEL_DIR,
    }

    model_name = st.radio("Choose model", list(MODEL_DIRS.keys()), horizontal=True)
    model_dir  = MODEL_DIRS[model_name]
    target_name = "bio_total" if "Biometric" in model_name else "enrol_total"

    required = [
        model_dir / "metrics.txt",
        model_dir / "district_metrics.csv",
        model_dir / "predictions_by_district_week.csv",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.warning(f"Missing required files in `{model_dir}`: {', '.join(missing)}")
        st.stop()

    metrics_text = (model_dir / "metrics.txt").read_text(encoding="utf-8")
    district_df  = pd.read_csv(model_dir / "district_metrics.csv")
    pred_df      = pd.read_csv(model_dir / "predictions_by_district_week.csv")
    pred_df["week_start"] = pd.to_datetime(pred_df["week_start"])

    def parse_simple_metrics(txt):
        vals = {}
        for line in txt.splitlines():
            if "=" not in line: continue
            k, v = line.split("=", 1)
            try: vals[k.strip().lower()] = float(v.strip().replace("%", ""))
            except: pass
        return vals

    parsed = parse_simple_metrics(metrics_text)
    overall_mae  = parsed.get("mae",  float(district_df["mae"].mean())  if "mae"  in district_df else 0.0)
    overall_rmse = parsed.get("rmse", float(district_df["rmse"].mean()) if "rmse" in district_df else 0.0)
    n_districts  = pred_df["district"].nunique()
    top_district = district_df.sort_values("mae").iloc[0]["district"] if not district_df.empty else "—"

    metric_strip(
        metric_card("Model", model_name, f"target = {target_name}", small=True),
        metric_card("Overall MAE",  f"{overall_mae:.3f}",  "lower is better"),
        metric_card("Overall RMSE", f"{overall_rmse:.3f}", "penalises larger errors"),
        metric_card("Districts", str(n_districts), "evaluated in test period"),
        metric_card("Best district", str(top_district)[:18], "lowest MAE", small=True),
    )

    def make_line_plot(x, y_act, y_pred, title, ylabel, bg="#f4f3ef"):
        fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=bg)
        ax.set_facecolor("#ffffff")
        ax.fill_between(x, y_act, alpha=0.08, color="#638cff")
        ax.plot(x, y_act,  linewidth=2.2, color="#638cff", label="Actual")
        ax.plot(x, y_pred, linewidth=2.2, color="#f97316", linestyle="--", label="Predicted")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10, color="#0f172a")
        ax.set_xlabel("Week", fontsize=9, color="#64748b")
        ax.set_ylabel(ylabel, fontsize=9, color="#64748b")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.grid(True, color="#f1f5f9", linewidth=0.8)
        ax.legend(fontsize=10, framealpha=0)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax.spines[sp].set_color("#e8e6e0")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return buf

    tabs = st.tabs(["National Trend", "Loss Curve", "District Leaderboard", "District Deep-Dive", "Raw Files"])

    with tabs[0]:
        weekly = pred_df.groupby("week_start")[["actual", "predicted"]].sum().reset_index()
        buf = make_line_plot(
            weekly["week_start"], weekly["actual"], weekly["predicted"],
            f"{model_name} — national weekly actual vs predicted", target_name,
        )
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(buf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        note(
            "National forecast aggregated across all districts. "
            "Close overlap means the model captures national movement well despite district-wise training.",
        )

    with tabs[1]:
        loss_path = model_dir / "loss_curve.png"
        if loss_path.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(loss_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            note("Blue = training loss · Orange = validation loss. "
                 "When validation stops improving, the model has extracted most useful signal.")
        else:
            st.info(f"`loss_curve.png` not found in `{model_dir}`. Other results are still available.")

    with tabs[2]:
        sort_metric = st.selectbox("Sort districts by", ["mae", "rmse"], key="stgcn_sort_metric")
        view_df = district_df.sort_values(sort_metric).copy()
        view_df[[c for c in ["mae", "rmse"] if c in view_df.columns]] = \
            view_df[[c for c in ["mae", "rmse"] if c in view_df.columns]].round(3)
        st.dataframe(view_df, hide_index=True, use_container_width=True)
        note("Lower MAE / RMSE = better forecast. Use this to identify stable vs irregular districts.")

    with tabs[3]:
        district_options = sorted(pred_df["district"].dropna().unique().tolist())
        selected_district = st.selectbox("Select district", district_options, key="stgcn_district_pick")
        sub = pred_df[pred_df["district"] == selected_district].sort_values("week_start")
        buf = make_line_plot(
            sub["week_start"], sub["actual"], sub["predicted"],
            f"{selected_district} — weekly actual vs predicted", target_name,
        )
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(buf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("District MAE",      f"{sub['abs_error'].mean():.3f}")
        c2.metric("Best week error",   f"{sub['abs_error'].min():.3f}")
        c3.metric("Worst week error",  f"{sub['abs_error'].max():.3f}")
        st.dataframe(
            sub.assign(week_start=sub["week_start"].dt.strftime("%Y-%m-%d")),
            hide_index=True, use_container_width=True,
        )

    with tabs[4]:
        with st.expander("metrics.txt"):
            st.code(metrics_text, language="text")
        with st.expander("district_metrics.csv"):
            st.dataframe(district_df, hide_index=True, use_container_width=True)
        with st.expander("predictions_by_district_week.csv"):
            disp = pred_df.copy()
            disp["week_start"] = disp["week_start"].dt.strftime("%Y-%m-%d")
            st.dataframe(disp, hide_index=True, use_container_width=True)
        st.download_button(
            "Download predictions_by_district_week.csv",
            data=pred_df.to_csv(index=False),
            file_name=f"{target_name}_predictions_by_district_week.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — TENSOR DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Tensor Decomposition":
    page_header(
        "Tensor Decomposition",
        "CP decomposition of weekly district tensors for biometric and enrolment data — latent patterns across time, districts and features",
    )

    MODEL_DIRS = {
        "Biometric Tensor": TENSOR_DIR / "biometric",
        "Enrolment Tensor": TENSOR_DIR / "enrolment",
    }

    tensor_name = st.radio("Choose tensor", list(MODEL_DIRS.keys()), horizontal=True)
    tensor_dir  = MODEL_DIRS[tensor_name]

    required = [
        tensor_dir / "reconstruction_metrics.txt",
        tensor_dir / "time_factors.csv",
        tensor_dir / "district_factors.csv",
        tensor_dir / "feature_factors.csv",
        tensor_dir / "component_strengths.csv",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.warning(f"Missing files in `{tensor_dir}`: {', '.join(missing)}")
        st.stop()

    metrics_text = (tensor_dir / "reconstruction_metrics.txt").read_text(encoding="utf-8")
    time_df     = pd.read_csv(tensor_dir / "time_factors.csv", index_col=0)
    district_df = pd.read_csv(tensor_dir / "district_factors.csv", index_col=0)
    feature_df  = pd.read_csv(tensor_dir / "feature_factors.csv", index_col=0)
    strength_df = pd.read_csv(tensor_dir / "component_strengths.csv")

    def parse_metrics(txt):
        vals = {}
        for line in txt.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip().lower()] = v.strip()
        return vals

    parsed   = parse_metrics(metrics_text)
    shape_str = parsed.get("shape", "—")
    rank_str  = parsed.get("rank", "—")
    fit_str   = parsed.get("approximate fit", "—")
    err_str   = parsed.get("relative reconstruction error", "—")

    metric_strip(
        metric_card("Tensor", tensor_name, "weekly district tensor", small=True),
        metric_card("Shape", shape_str, "T × N × C", small=True),
        metric_card("Rank", rank_str, "number of latent components"),
        metric_card("Approx. Fit", fit_str, "higher is better"),
        metric_card("Reconstruction Error", err_str, "lower is better"),
        metric_card("Components", str(len(strength_df)), "interpretable patterns"),
    )

    is_bio   = tensor_name == "Biometric Tensor"
    is_enrol = tensor_name == "Enrolment Tensor"

    tabs = st.tabs(["Time Factors", "District Patterns", "Feature Patterns", "Top Districts", "Raw Files"])

    # ── Bio/Enrol interpretation blocks (condensed inline) ───────────────────
    def bio_time_interp():
        st.markdown(INTERP_CSS, unsafe_allow_html=True)
        st.markdown("<div class='interp-section'><div class='interp-header'>📅 How to Read the Time Factors Plot</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='interp-card'><div class='interp-card-header'>
            <span class='comp-badge'>C1</span>
            <span class='comp-title'>Monthly batch spikes — Mar to Jul 2025</span></div>
            <div class='comp-body'>Sharp peaks every ~4 weeks Mar–Jul 2025, then smoother from Aug. Suggests
            <b>monthly Aadhaar data release cycles</b> — bulk records pushed in batches.</div></div>
        <div class='interp-card'><div class='interp-card-header'>
            <span class='comp-badge' style='background:#d97706;'>C2–C3</span>
            <span class='comp-title'>Stable negative baseline</span></div>
            <div class='comp-body'>Flat negative lines throughout (~−20 and −23). These are
            <b>constant background offsets</b> — not declining activity.</div></div>
        <div class='interp-card'><div class='interp-card-header'>
            <span class='comp-badge' style='background:#059669;'>C4–C7</span>
            <span class='comp-title'>Near-zero — temporally diffuse patterns</span></div>
            <div class='comp-body'>Signal lives almost entirely in the <b>district and feature dimensions</b> — 
            these patterns hold consistently across the entire study period.</div></div>
        <div class='interp-note'><b>Key takeaway:</b> Time axis dominated by C1's batch-upload rhythm.
        All other components are temporally stable.</div></div>
        """, unsafe_allow_html=True)

    def enrol_time_interp():
        st.markdown(INTERP_CSS, unsafe_allow_html=True)
        st.markdown("<div class='interp-section'><div class='interp-header'>📅 Time Factors — Enrolment Tensor</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='interp-card'><div class='interp-card-header'>
            <span class='comp-badge'>C1</span>
            <span class='comp-title'>Monthly spikes Mar–Jul, then persistent plateau</span></div>
            <div class='comp-body'>Unlike biometric, enrolment C1 stays elevated post-July (~40–70),
            indicating <b>continued bulk enrolment pushes in H2 2025</b>.</div></div>
        <div class='interp-card'><div class='interp-card-header'>
            <span class='comp-badge' style='background:#d97706;'>C2</span>
            <span class='comp-title'>Secondary wave — co-occurs with C1 bursts</span></div>
            <div class='comp-body'>Spikes alongside C1 in Apr, May, Jul. Captures a <b>secondary
            enrolment wave</b> — same temporal trigger, different demographic signature.</div></div>
        <div class='interp-card'><div class='interp-card-header'>
            <span class='comp-badge' style='background:#7c3aed;'>C4</span>
            <span class='comp-title'>Steady positive baseline (~9)</span></div>
            <div class='comp-body'>Constant ~9 throughout — the <b>steady-state enrolment background</b>
            for districts that enrol consistently rather than in batches.</div></div>
        <div class='interp-note'><b>Key takeaway:</b> Enrolment shows a richer temporal story — two
        components spike in batch months, C4 provides a steady baseline, C3 is a stable offset.
        July 2025 is the strongest event in the dataset.</div></div>
        """, unsafe_allow_html=True)

    with tabs[0]:
        img_path = tensor_dir / "time_factors.png"
        if img_path.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(img_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.line_chart(time_df)
        note("Each line = one latent component across weeks. Peaks = weeks where that pattern is strongly active.")
        if is_bio:   bio_time_interp()
        if is_enrol: enrol_time_interp()

    with tabs[1]:
        img_path = tensor_dir / "district_component_heatmap.png"
        if img_path.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(img_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        district_component = st.selectbox(
            "Component for district ranking", district_df.columns.tolist(), key="tensor_district_component"
        )
        top_d = district_df[district_component].abs().sort_values(ascending=False).head(25).reset_index()
        top_d.columns = ["district", "loading"]
        st.dataframe(top_d, hide_index=True, use_container_width=True)
        note("Large absolute loading = that district strongly expresses this latent pattern.")

    with tabs[2]:
        img_path = tensor_dir / "feature_component_heatmap.png"
        if img_path.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(img_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        feature_component = st.selectbox(
            "Component for feature ranking", feature_df.columns.tolist(), key="tensor_feature_component"
        )
        top_f = feature_df[feature_component].abs().sort_values(ascending=False).reset_index()
        top_f.columns = ["feature", "loading"]
        st.dataframe(top_f, hide_index=True, use_container_width=True)
        note("Feature factor matrix tells us which variables define each component.")

    with tabs[3]:
        file_path = tensor_dir / "top_districts_by_component.csv"
        if file_path.exists():
            top_df = pd.read_csv(file_path)
            st.dataframe(top_df, hide_index=True, use_container_width=True)
        else:
            st.info("`top_districts_by_component.csv` not found.")

        if is_bio:
            st.markdown(INTERP_CSS, unsafe_allow_html=True)
            st.markdown("<div class='interp-section'><div class='interp-header'>🏆 Component Summary</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='interp-card'><div class='comp-meta'>
                <div class='comp-meta-item'><b>C1 — Urban volume</b><br>Pune · Mumbai · Nashik<br>Driven by: absolute bio counts</div>
                <div class='comp-meta-item'><b>C2 — Young child dependency (UP)</b><br>Sitapur · Kaushambi · Aligarh<br>Driven by: age_5_ratio, dependency_ratio</div>
                <div class='comp-meta-item'><b>C3 — Teen ratio (Nagaland)</b><br>Zunheboto · Longleng · Kiphire<br>Driven by: age_17_ratio</div>
                <div class='comp-meta-item'><b>C4 — Dependency burden (NE+UP)</b><br>Kaushambi · Champhai · Saiha<br>Driven by: dependency_ratio</div>
            </div></div>
            <div class='interp-card'><div class='comp-meta'>
                <div class='comp-meta-item'><b>C5 — Rural youth (AP+Raj)</b><br>Kurnool · Banswara · Dungarpur<br>Driven by: all ratio features</div>
                <div class='comp-meta-item'><b>C6 — Remote scaling ⚠️</b><br>Kargil · Upper Siang · Dibang<br>Driven by: small-denominator extremes</div>
                <div class='comp-meta-item'><b>C7 — Frontier (J&K/AR)</b><br>Ramban · Tawang · Kargil<br>Driven by: dependency_ratio</div>
            </div></div>
            <div class='interp-note'><b>Overall fit:</b> 68.5% of tensor variance captured at rank 7.
            7 components describe the main demographic axes along which Indian districts differ.</div></div>
            """, unsafe_allow_html=True)

        if is_enrol:
            st.markdown(INTERP_CSS, unsafe_allow_html=True)
            st.markdown("<div class='interp-section'><div class='interp-header'>🏆 Component Summary — Enrolment Tensor</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='interp-card'><div class='comp-meta'>
                <div class='comp-meta-item'><b>C1 — Bihar child volume</b><br>Pashchim Champaran · Gaya · Patna<br>Driven by: age_5_17, enrol_total</div>
                <div class='comp-meta-item'><b>C2 — Meghalaya adult surge</b><br>East Khasi Hills · W Khasi Hills<br>Driven by: age_18_greater</div>
                <div class='comp-meta-item'><b>C3 — Minor ratio (UP/MH)</b><br>Purba Champaran · Nashik · Hardoi<br>Driven by: enrol_minor_ratio</div>
                <div class='comp-meta-item'><b>C4 — Adult ratio anomaly (NE+PB)</b><br>W Khasi Hills · Dibrugarh · Kapurthala<br>Driven by: enrol_adult_ratio</div>
            </div></div>
            <div class='interp-card'><div class='comp-meta'>
                <div class='comp-meta-item'><b>C5 — WB infant enrolments</b><br>Murshidabad · S24P · N24P<br>Driven by: age_0_5</div>
                <div class='comp-meta-item'><b>C6 — Population size ⚠️</b><br>Pashchim Champaran · Hyderabad<br>Driven by: raw headcount (millions)</div>
                <div class='comp-meta-item'><b>C7 — Pan-India minor floor</b><br>Uniform 0.321 loading<br>Driven by: enrol_minor_ratio (background)</div>
            </div></div>
            <div class='interp-note'><b>Overall fit:</b> 65.8% of tensor variance at rank 7. Enrolment is structured by
            <b>age-group mix</b>, <b>geography</b> (Bihar, Meghalaya, West Bengal) and <b>batch timing</b>.</div></div>
            """, unsafe_allow_html=True)

    with tabs[4]:
        with st.expander("reconstruction_metrics.txt"):
            st.code(metrics_text, language="text")
        with st.expander("component_strengths.csv"):
            st.dataframe(strength_df, hide_index=True, use_container_width=True)
        with st.expander("time_factors.csv"):
            st.dataframe(time_df, use_container_width=True)
        with st.expander("district_factors.csv"):
            st.dataframe(district_df, use_container_width=True)
        with st.expander("feature_factors.csv"):
            st.dataframe(feature_df, use_container_width=True)
        rank_path = tensor_dir / "feature_rankings_by_component.csv"
        if rank_path.exists():
            with st.expander("feature_rankings_by_component.csv"):
                st.dataframe(pd.read_csv(rank_path), hide_index=True, use_container_width=True)