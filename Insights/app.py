import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
st.set_page_config(page_title="Aadhaar Data Mining", page_icon="🗺️",
                   layout="wide", initial_sidebar_state="expanded")

CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:#0f1117;border-right:1px solid #1e2130;}
[data-testid="stSidebar"] *{color:#c8cad4 !important;}
[data-testid="stAppViewContainer"]{background:#f9f8f6;}
[data-testid="stAppViewContainer"] > .main > div{padding-top:1.5rem;}
        "<div style='font-size:1.3rem;color:#ffffff;line-height:1.3;margin-bottom:4px;'>"
.section-sub{font-size:14px;color:#6b7280;margin-bottom:1.8rem;font-weight:300;}
.metric-row{display:flex;gap:14px;margin-bottom:1.8rem;flex-wrap:wrap;}
.metric-card{flex:1;min-width:140px;background:white;border:1px solid #e5e7eb;border-radius:12px;padding:18px 20px;}
.metric-label{font-size:11px;font-weight:500;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;}
        "<div style='font-size:1.3rem;color:#ffffff;line-height:1.3;margin-bottom:4px;'>"
.metric-note{font-size:12px;color:#6b7280;}
.insight-card{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:16px 20px;margin-bottom:12px;}
.insight-cluster{font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;}
.insight-title{font-size:15px;font-weight:500;color:#1a1a2e;margin-bottom:6px;}
.insight-body{font-size:13px;color:#6b7280;line-height:1.65;}
.img-frame{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:12px;margin-bottom:12px;}
.note-box{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:16px 20px;font-size:13px;color:#6b7280;line-height:1.7;margin-top:12px;}
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

CLUSTER_DIR = Path("clustering_output")
STATE_DIR      = Path("state_output")
TIMESERIES_DIR = Path("timeseries_output")
SPATIAL_DIR    = Path("spatial_output")
TABLE_DIR  = Path("table_output")
STGCN_DIR  = Path("../STCGN/stgcn_output")
STATE_DIR      = Path("state_output")
TIMESERIES_DIR = Path("timeseries_output")
SPATIAL_DIR    = Path("spatial_output")
TABLE_DIR  = Path("table_output")


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:8px 0 20px'>"
        "<div style='font-size:1.3rem;color:#ffffff;line-height:1.3;margin-bottom:4px;'>"
        "color:#ffffff;line-height:1.3;margin-bottom:4px;'>Aadhaar<br>Data Mining</div>"
        "<div style='font-size:11px;color:#6b7280;letter-spacing:.05em;"
        "text-transform:uppercase;'>Research Dashboard</div></div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio("Navigation",
        ["Clustering Analysis", "Table Analysis", "State Comparison", "Time Series Trends", "District Deep-Dive", "STGCN Results"],
        label_visibility="collapsed")
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px;color:#4b5563;padding:12px;"
        "background:#1a1f2e;border-radius:8px;line-height:1.7;'>"
        "<b style='color:#9ca3af'>Data source</b><br>"
        "UIDAI Aadhaar open data<br>Apr–Jul 2025<br>"
        "775 districts · 3 tables<br>~5M records</div>",
        unsafe_allow_html=True
    )

def img(path):
    if Path(path).exists():
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(str(path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"{Path(path).name} not found — run the analysis script first.")

def note(title, body):
    st.markdown(
        f"<div class='note-box'><b style='color:#1a1a2e;'>{title}</b>"
        f"<br><br>{body}</div>",
        unsafe_allow_html=True
    )

def icard(color, cluster_label, title, body):
    st.markdown(
        f"<div class='insight-card' style='border-left:3px solid {color};'>"
        f"<div class='insight-cluster' style='color:{color};'>{cluster_label}</div>"
        f"<div class='insight-title'>{title}</div>"
        f"<div class='insight-body'>{body}</div></div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════
if page == "Clustering Analysis":
    st.markdown("<div class='section-title'>District Clustering Analysis</div>"
                "<div class='section-sub'>K-Means and DBSCAN applied to engineered Aadhaar adoption features "
                "across 775 Indian districts · Apr–Jul 2025</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='metric-row'>"
        "<div class='metric-card'><div class='metric-label'>Districts</div>"
        "<div class='metric-value'>775</div><div class='metric-note'>28 states + UTs</div></div>"
        "<div class='metric-card'><div class='metric-label'>Features used</div>"
        "<div class='metric-value'>18</div><div class='metric-note'>ratios, ranks, growth</div></div>"
        "<div class='metric-card'><div class='metric-label'>K-Means K</div>"
        "<div class='metric-value'>5</div><div class='metric-note'>silhouette = 0.303</div></div>"
        "<div class='metric-card'><div class='metric-label'>DBSCAN noise</div>"
        "<div class='metric-value'>40</div><div class='metric-note'>outlier districts</div></div>"
        "<div class='metric-card'><div class='metric-label'>Time steps</div>"
        "<div class='metric-value'>4</div><div class='metric-note'>monthly snapshots</div></div>"
        "</div>", unsafe_allow_html=True)

    t1,t2,t3,t4,t5 = st.tabs(["K-Means map","DBSCAN map","Cluster profiles","Elbow/silhouette","PCA scatter"])

    with t1:
        c1,c2 = st.columns([2,1])
        with c1:
            img(CLUSTER_DIR/"kmeans_choropleth.png")
            h = CLUSTER_DIR/"kmeans_choropleth.html"
            if h.exists():
                st.info(f"Interactive version → open `{h}` in your browser to hover districts.")
        with c2:
            st.markdown("<div style='font-size:13px;font-weight:500;color:#1a1a2e;margin-bottom:10px;'>Cluster legend</div>", unsafe_allow_html=True)
            for n,col,title,desc in [
                ("0","#1D9E75","Mainstream coverage","Balanced ratios. Stable trends. South India, Rajasthan periphery, Northeast."),
                ("1","#534AB7","High-dependency zones","High child-to-adult ratio. UP, Bihar, MP — younger demographics."),
                ("2","#D85A30","High growth momentum","Elevated daily_pct_change. Rapid enrolment spikes from local campaigns."),
                ("3","#BA7517","Adult-dominant pattern","High adult ratio. South India, Gujarat — mature Aadhaar saturation."),
                ("4","#D4537E","Restricted / border zones","J&K, Ladakh. Low ratios, high volatility — different administration."),
            ]:
                icard(col, f"Cluster {n}", title, desc)

    with t2:
        c1,c2 = st.columns([2,1])
        with c1: img(CLUSTER_DIR/"dbscan_choropleth.png")
        with c2:
            st.markdown("<div style='font-size:13px;font-weight:500;color:#1a1a2e;margin-bottom:12px;'>What DBSCAN reveals</div>", unsafe_allow_html=True)
            icard("#534AB7","Cluster 1 · 726 districts","The mainstream","94% of districts share one dense cluster — uniform Aadhaar adoption across India.")
            icard("#1D9E75","Cluster 0 · 9 districts","Metro outlier cluster","9 districts distinct from mainstream — likely high-volume urban centres.")
            icard("#9ca3af","Noise · 40 districts","True outliers","40 districts fit no cluster. Check cluster_summary.csv where dbscan_cluster == -1.")

    with t3:
        img(CLUSTER_DIR/"cluster_profiles_kmeans.png")
        csv = CLUSTER_DIR/"cluster_summary.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            counts = (df.groupby('kmeans_cluster')['district'].count().reset_index()
                        .rename(columns={'district':'Districts','kmeans_cluster':'Cluster'}))
            counts['Cluster'] = counts['Cluster'].apply(lambda x: f"Cluster {x}")
            st.dataframe(counts, hide_index=True, use_container_width=False)
            with st.expander("Browse all districts"):
                search = st.text_input("Search district or state","")
                cols = [c for c in ['district','state','kmeans_cluster','dbscan_cluster'] if c in df.columns]
                filtered = df[cols]
                if search:
                    mask = (filtered['district'].str.contains(search,case=False,na=False) |
                            filtered['state'].str.contains(search,case=False,na=False))
                    filtered = filtered[mask]
                st.dataframe(filtered, hide_index=True, use_container_width=True)

    with t4:
        img(CLUSTER_DIR/"elbow_silhouette.png")
        note("Choosing K",
             "The <b>elbow curve</b> shows inertia dropping as K increases — the bend marks diminishing returns. "
             "The <b>silhouette score</b> measures cluster separation (1.0 = perfect). We used <b>K=5</b> for "
             "a balance of statistical optimality and geographic interpretability.")

    with t5:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**K-Means — PCA space**")
            img(CLUSTER_DIR/"pca_kmeans.png")
        with c2:
            st.markdown("**DBSCAN — PCA space**")
            img(CLUSTER_DIR/"pca_dbscan.png")
        note("Reading the PCA scatter",
             "Each dot is one district projected from 18 features to 2D. PCA captures 42.8% of total variance — "
             "overlap in this view doesn't mean bad clusters; separation exists in higher dimensions.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — STATE COMPARISON
# ══════════════════════════════════════════════════════════════════════════
elif page == "State Comparison":
    st.markdown("<div class='section-title'>State-level Comparison</div>"
                "<div class='section-sub'>Rankings, age ratios, growth momentum and dependency patterns "
                "across all Indian states · Apr–Jul 2025</div>", unsafe_allow_html=True)

    csv_path = STATE_DIR/"state_summary.csv"
    if not csv_path.exists():
        st.warning("Run state_comparison.py first to generate the charts and data.")
        st.code("python state_comparison.py --db ../database/aadhar.duckdb")
        st.stop()

    sdf = pd.read_csv(csv_path)
    top_enrol  = sdf.loc[sdf['enrol_total'].idxmax(),'state']
    top_growth = sdf.loc[sdf['avg_enrol_growth'].idxmax(),'state']
    top_dep    = sdf.loc[sdf['avg_dependency_ratio'].idxmax(),'state']
    top_adult  = sdf.loc[sdf['avg_adult_ratio'].idxmax(),'state']

    st.markdown(
        f"<div class='metric-row'>"
        f"<div class='metric-card'><div class='metric-label'>States analysed</div>"
        f"<div class='metric-value'>{len(sdf)}</div><div class='metric-note'>+ union territories</div></div>"
        f"<div class='metric-card'><div class='metric-label'>Highest enrolment</div>"
        f"<div class='metric-value' style='font-size:1.2rem;'>{top_enrol[:18]}</div>"
        f"<div class='metric-note'>total volume Apr–Jul</div></div>"
        f"<div class='metric-card'><div class='metric-label'>Fastest growing</div>"
        f"<div class='metric-value' style='font-size:1.2rem;'>{top_growth[:18]}</div>"
        f"<div class='metric-note'>highest avg daily growth %</div></div>"
        f"<div class='metric-card'><div class='metric-label'>Highest dependency</div>"
        f"<div class='metric-value' style='font-size:1.2rem;'>{top_dep[:18]}</div>"
        f"<div class='metric-note'>child-to-adult ratio</div></div>"
        f"<div class='metric-card'><div class='metric-label'>Most adult-saturated</div>"
        f"<div class='metric-value' style='font-size:1.2rem;'>{top_adult[:18]}</div>"
        f"<div class='metric-note'>highest adult enrol ratio</div></div>"
        f"</div>", unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6 = st.tabs([
        "Enrolment ranking","Adult vs minor","Growth momentum",
        "Dependency heatmap","Size vs growth","Data table"])

    with t1:
        c1,c2 = st.columns([2,1])
        with c1: img(STATE_DIR/"state_enrolment_bar.png")
        with c2:
            note("What this shows",
                 "Total Aadhaar enrolment summed across all districts within each state "
                 "over Apr–Jul 2025. <b style='color:#534AB7;'>Purple bars</b> = top 5 states by volume. "
                 "States with more districts appear higher — use the other tabs for size-agnostic comparisons.")

    with t2:
        c1,c2 = st.columns([2,1])
        with c1: img(STATE_DIR/"state_adult_vs_minor.png")
        with c2:
            note("Adult vs minor ratio",
                 "Split between minor (age 0–17) and adult (18+) enrolments as a proportion of the state total. "
                 "Sorted by adult ratio.<br><br>"
                 "<b>High minor ratio</b> = younger population or ongoing child Aadhaar push.<br><br>"
                 "<b>High adult ratio</b> = mature, near-saturated adult coverage.")

    with t3:
        c1,c2 = st.columns([2,1])
        with c1: img(STATE_DIR/"state_growth_bar.png")
        with c2:
            note("Growth momentum",
                 "Average daily % change in enrolment across all districts in each state. "
                 "<b style='color:#D85A30;'>Orange</b> = above-median growth. "
                 "<b style='color:#9ca3af;'>Grey</b> = below-median. "
                 "The dashed purple line is the national median. "
                 "States above it had active enrolment drives during the period.")

    with t4:
        c1,c2 = st.columns([2,1])
        with c1: img(STATE_DIR/"state_dependency_heatmap.png")
        with c2:
            note("Dependency heatmap",
                 "Child-to-adult biometric dependency ratio for each state across the 4 monthly time steps.<br><br>"
                 "<b>Darker red</b> = higher proportion of children vs adults.<br>"
                 "<b>Lighter yellow</b> = adult-dominant.<br><br>"
                 "Consistent dark across months = structural demographic. "
                 "Changing colours = shifting enrolment pattern.")

    with t5:
        img(STATE_DIR/"state_scatter.png")
        note("How to read this chart",
             "X = number of districts (state size). Y = avg daily growth %. "
             "Bubble size = total enrolment. Colour = adult ratio (green = high adult coverage).<br><br>"
             "<b>Top-left</b>: small states, high growth — campaigns working efficiently.<br>"
             "<b>Bottom-right</b>: large states, slower growth — scale challenge.")

    with t6:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            search_state = st.text_input("Search state","")
        with col_f2:
            sort_col = st.selectbox("Sort by",[
                "enrol_total","avg_enrol_growth","avg_adult_ratio",
                "avg_minor_ratio","avg_dependency_ratio","district_count"])
        disp = sdf.copy()
        disp[disp.select_dtypes('float').columns] = disp.select_dtypes('float').round(4)
        if search_state:
            disp = disp[disp['state'].str.contains(search_state, case=False, na=False)]
        disp = disp.sort_values(sort_col, ascending=False)
        st.dataframe(disp, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", disp.to_csv(index=False),
                           "state_summary.csv","text/csv")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — STATE COMPARISON
# ══════════════════════════════════════════════════════════════════════════

elif page == "State Comparison":

    st.markdown("""
    <div class='section-title'>State-level Comparison</div>
    <div class='section-sub'>
        Rankings, age ratios, growth momentum and dependency patterns
        across all Indian states · Apr–Jul 2025
    </div>
    """, unsafe_allow_html=True)

    csv_path = STATE_DIR / "state_summary.csv"
    if not csv_path.exists():
        st.warning("Run state_comparison.py first to generate the charts and data.")
        st.code("python state_comparison.py --db ../database/aadhar.duckdb")
        st.stop()

    sdf = pd.read_csv(csv_path)
    top_enrol  = sdf.loc[sdf["enrol_total"].idxmax(), "state"]
    top_growth = sdf.loc[sdf["avg_enrol_growth"].idxmax(), "state"]
    top_dep    = sdf.loc[sdf["avg_dependency_ratio"].idxmax(), "state"]
    top_adult  = sdf.loc[sdf["avg_adult_ratio"].idxmax(), "state"]

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='metric-label'>States analysed</div>
            <div class='metric-value'>{len(sdf)}</div>
            <div class='metric-note'>+ union territories</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Highest enrolment</div>
            <div class='metric-value' style='font-size:1.2rem;'>{top_enrol[:18]}</div>
            <div class='metric-note'>total volume Apr–Jul</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Fastest growing</div>
            <div class='metric-value' style='font-size:1.2rem;'>{top_growth[:18]}</div>
            <div class='metric-note'>highest avg daily growth %</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Highest dependency</div>
            <div class='metric-value' style='font-size:1.2rem;'>{top_dep[:18]}</div>
            <div class='metric-note'>child-to-adult ratio</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Most adult-saturated</div>
            <div class='metric-value' style='font-size:1.2rem;'>{top_adult[:18]}</div>
            <div class='metric-note'>highest adult enrol ratio</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Helper to show image or warning
    def show_img(path):
        p = Path(path)
        if p.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(p), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"{p.name} not found — run state_comparison.py first.")

    def note_box(title, body):
        st.markdown(f"""
        <div style='background:white;border:1px solid #e5e7eb;border-radius:12px;
                    padding:16px 20px;font-size:13px;color:#6b7280;line-height:1.7;'>
            <b style='color:#1a1a2e;'>{title}</b><br><br>{body}
        </div>
        """, unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Enrolment ranking",
        "Adult vs minor",
        "Growth momentum",
        "Dependency heatmap",
        "Size vs growth",
        "Data table",
    ])

    with t1:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(STATE_DIR / "state_enrolment_bar.png")
        with c2:
            note_box("What this shows",
                "Total Aadhaar enrolment summed across all districts within each state "
                "over Apr–Jul 2025. <b style='color:#534AB7;'>Purple bars</b> = top 5 states by volume. "
                "Use the other tabs for size-agnostic comparisons.")

    with t2:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(STATE_DIR / "state_adult_vs_minor.png")
        with c2:
            note_box("Adult vs minor ratio",
                "Split between minor (age 0–17) and adult (18+) enrolments as a proportion of state total. "
                "Sorted by adult ratio.<br><br>"
                "<b>High minor ratio</b> = younger population or ongoing child Aadhaar push.<br><br>"
                "<b>High adult ratio</b> = mature, near-saturated adult coverage.")

    with t3:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(STATE_DIR / "state_growth_bar.png")
        with c2:
            note_box("Growth momentum",
                "Average daily % change in enrolment across all districts in each state. "
                "<b style='color:#D85A30;'>Orange</b> = above-median growth. "
                "<b style='color:#9ca3af;'>Grey</b> = below-median. "
                "The dashed purple line is the national median.")

    with t4:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(STATE_DIR / "state_dependency_heatmap.png")
        with c2:
            note_box("Dependency heatmap",
                "Child-to-adult biometric dependency ratio for each state across the 4 monthly time steps.<br><br>"
                "<b>Darker red</b> = higher proportion of children vs adults.<br>"
                "<b>Lighter yellow</b> = adult-dominant.<br><br>"
                "Consistent dark across months = structural demographic pattern. "
                "Colour changing across months = enrolment wave shifting.")

    with t5:
        show_img(STATE_DIR / "state_scatter.png")
        note_box("How to read this chart",
            "X = number of districts (state size). Y = avg daily growth %. "
            "Bubble size = total enrolment. Colour = adult ratio (green = high adult coverage).<br><br>"
            "<b>Top-left</b>: small states, high growth — campaigns working efficiently.<br>"
            "<b>Bottom-right</b>: large states, slower growth — scale challenge.")

    with t6:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            search_state = st.text_input("Search state", "")
        with col_f2:
            sort_col = st.selectbox("Sort by", [
                "enrol_total", "avg_enrol_growth", "avg_adult_ratio",
                "avg_minor_ratio", "avg_dependency_ratio", "district_count"
            ])
        disp = sdf.copy()
        float_cols = disp.select_dtypes("float").columns
        disp[float_cols] = disp[float_cols].round(4)
        if search_state:
            disp = disp[disp["state"].str.contains(search_state, case=False, na=False)]
        disp = disp.sort_values(sort_col, ascending=False)
        st.dataframe(disp, hide_index=True, use_container_width=True)
        st.download_button(
            label="Download CSV",
            data=disp.to_csv(index=False),
            file_name="state_summary.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — TIME SERIES TRENDS
# ══════════════════════════════════════════════════════════════════════════

elif page == "Time Series Trends":

    st.markdown("""
    <div class='section-title'>Time Series Trends</div>
    <div class='section-sub'>
        Aadhaar enrolment patterns over 70 dates · Apr–Dec 2025 ·
        national daily totals, state trends, heatmap, growth and volatility
    </div>
    """, unsafe_allow_html=True)

    # Check outputs exist
    if not (TIMESERIES_DIR / "ts_national.png").exists():
        st.warning("Time series charts not found. Run time_series.py first.")
        st.code("python time_series.py --db ../database/aadhar.duckdb")
        st.stop()

    # ── Metric row ─────────────────────────────────────────────────────
    st.markdown("""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='metric-label'>Date range</div>
            <div class='metric-value' style='font-size:1.2rem;'>Apr–Dec</div>
            <div class='metric-note'>2025 · 9 months</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Total time steps</div>
            <div class='metric-value'>70</div>
            <div class='metric-note'>common across 3 tables</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Granularity</div>
            <div class='metric-value' style='font-size:1.2rem;'>Daily</div>
            <div class='metric-note'>Sep–Dec · Monthly Apr–Jul</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Metric tracked</div>
            <div class='metric-value' style='font-size:1.2rem;'>Enrol</div>
            <div class='metric-note'>+ biometric overlay</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Smoothing</div>
            <div class='metric-value' style='font-size:1.2rem;'>7-day</div>
            <div class='metric-note'>rolling average applied</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    def show_img(path):
        p = Path(path)
        if p.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(p), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"{p.name} not found.")

    def note_box(title, body):
        st.markdown(
            f"<div style='background:white;border:1px solid #e5e7eb;border-radius:12px;"
            f"padding:16px 20px;font-size:13px;color:#6b7280;line-height:1.7;'>"
            f"<b style='color:#1a1a2e;'>{title}</b><br><br>{body}</div>",
            unsafe_allow_html=True
        )

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "National trend",
        "Top 10 states",
        "State heatmap",
        "Monthly growth",
        "Volatility",
        "Raw data",
    ])

    # ── Tab 1: National daily total ─────────────────────────────────────
    with t1:
        show_img(TIMESERIES_DIR / "ts_national.png")
        note_box("National enrolment trend",
            "Total Aadhaar enrolment summed across all states per day. "
            "Light green bars show raw daily values. The solid green line is a "
            "7-day rolling average to smooth day-to-day noise. "
            "The dashed purple line shows biometric enrolment for comparison.<br><br>"
            "The gap between Apr–Jul and Sep is expected — your dataset has monthly "
            "snapshots for Apr–Jul and daily records from Sep onwards. "
            "Vertical grey lines mark the start of each calendar month.")

    # ── Tab 2: Top 10 states ─────────────────────────────────────────────
    with t2:
        show_img(TIMESERIES_DIR / "ts_top10_states.png")
        note_box("Top 10 states by total enrolment",
            "Each line represents one state's enrolment over time, smoothed with a "
            "3-point rolling average. Dots mark actual recorded data points.<br><br>"
            "States with more districts naturally show higher absolute volumes — "
            "use the Growth tab to compare percentage changes instead.<br><br>"
            "Crossing lines indicate states overtaking each other in enrolment pace.")

    # ── Tab 3: Heatmap ───────────────────────────────────────────────────
    with t3:
        show_img(TIMESERIES_DIR / "ts_heatmap.png")
        note_box("State × date intensity heatmap",
            "Each row is a state, each column is a date. Colour intensity shows "
            "enrolment volume <b>normalised per state</b> (each state's max = 1.0) "
            "so that small and large states are visually comparable.<br><br>"
            "<b>Dark red columns</b> = unusually high enrolment day across many states — "
            "likely a national Aadhaar campaign or reporting batch.<br><br>"
            "<b>Dark red rows</b> = states consistently at peak enrolment throughout the period.")

    # ── Tab 4: Monthly growth ────────────────────────────────────────────
    with t4:
        show_img(TIMESERIES_DIR / "ts_monthly_growth.png")
        note_box("Month-over-month growth rate",
            "Grouped bars show the % change in enrolment from one month to the next "
            "for the top 20 states by total volume.<br><br>"
            "A bar above 0 = enrolment increased that month. Below 0 = declined.<br><br>"
            "Large positive bars in Sep reflect the transition from monthly to daily "
            "data — the apparent spike is partly a data density effect, not purely "
            "a real enrolment jump. Use this chart to compare <i>relative</i> growth across states.")

    # ── Tab 5: Volatility ────────────────────────────────────────────────
    with t5:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(TIMESERIES_DIR / "ts_volatility.png")
        with c2:
            note_box("Enrolment volatility",
                "Average 7-day standard deviation in enrolment per state — "
                "a measure of how irregular or spiky the enrolment pattern is.<br><br>"
                "<b style='color:#D85A30;'>Orange bars</b> = above-median volatility — "
                "these states have uneven enrolment, suggesting campaign-driven spikes "
                "rather than a steady flow.<br><br>"
                "<b style='color:#9ca3af;'>Grey bars</b> = steady, predictable enrolment. "
                "Low volatility states have consistent administrative processes.")

    # ── Tab 6: Raw data ──────────────────────────────────────────────────
    with t6:
        csv_path = TIMESERIES_DIR / "ts_data.csv"
        if csv_path.exists():
            tsdf = pd.read_csv(csv_path, index_col=0)
            st.markdown(f"**State × date enrolment pivot  "
                        f"({len(tsdf)} states × {len(tsdf.columns)} dates)**")

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                search_ts = st.text_input("Search state", "", key="ts_search")
            with col_f2:
                sort_ts = st.selectbox("Sort by total", ["Descending","Ascending"])

            disp = tsdf.copy()
            if search_ts:
                disp = disp[disp.index.str.contains(search_ts, case=False, na=False)]
            disp["_total"] = disp.sum(axis=1)
            disp = disp.sort_values("_total", ascending=(sort_ts=="Ascending"))
            disp = disp.drop(columns=["_total"])

            st.dataframe(disp, use_container_width=True)
            st.download_button(
                label="Download ts_data.csv",
                data=disp.to_csv(),
                file_name="ts_data.csv",
                mime="text/csv",
            )
        else:
            st.warning("ts_data.csv not found in timeseries_output/")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — SPATIAL AUTOCORRELATION
# ══════════════════════════════════════════════════════════════════════════

elif page == "Spatial Autocorrelation":

    st.markdown("""
    <div class='section-title'>Spatial Autocorrelation</div>
    <div class='section-sub'>
        Moran\'s I analysis — does high Aadhaar enrolment cluster geographically
        or is it randomly distributed across Indian districts?
    </div>
    """, unsafe_allow_html=True)

    if not (SPATIAL_DIR / "moran_scatter.png").exists():
        st.warning("Spatial analysis outputs not found. Run spatial_autocorr.py first.")
        st.code(
            "python spatial_autocorr.py "
            "--db ../database/aadhar.duckdb "
            "--shp Adjacency_marix/2011_Dist.shp "
            "--w adjacency_output/W_combined.csv"
        )
        st.stop()

    # ── Read report for metrics ─────────────────────────────────────────
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

    # ── Metric row ──────────────────────────────────────────────────────
    interp = "Spatially clustered" if global_I and global_I > 0 else "Spatially dispersed"
    sig_label = ""
    if global_p is not None:
        sig_label = "p < 0.001 ***" if global_p < 0.001 else "p < 0.01 **" if global_p < 0.01 else "p < 0.05 *" if global_p < 0.05 else "not significant"

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='metric-label'>Global Moran\'s I</div>
            <div class='metric-value'>{f"{global_I:.4f}" if global_I else "—"}</div>
            <div class='metric-note'>{interp}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Significance</div>
            <div class='metric-value' style='font-size:1.1rem;'>{sig_label}</div>
            <div class='metric-note'>permutation test (999 runs)</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Hot spots (HH)</div>
            <div class='metric-value' style='color:#D85A30;'>{hh_count}</div>
            <div class='metric-note'>high surrounded by high</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Cold spots (LL)</div>
            <div class='metric-value' style='color:#378ADD;'>{ll_count}</div>
            <div class='metric-note'>low surrounded by low</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Spatial outliers</div>
            <div class='metric-value'>{hl_count + lh_count}</div>
            <div class='metric-note'>HL + LH districts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    def show_img(path):
        p = Path(path)
        if p.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(p), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"{p.name} not found.")

    def note_box(title, body):
        st.markdown(
            f"<div style='background:white;border:1px solid #e5e7eb;border-radius:12px;"
            f"padding:16px 20px;font-size:13px;color:#6b7280;line-height:1.7;'>"
            f"<b style='color:#1a1a2e;'>{title}</b><br><br>{body}</div>",
            unsafe_allow_html=True
        )

    t1, t2, t3, t4, t5 = st.tabs([
        "LISA map",
        "Moran scatter",
        "All features",
        "Report",
        "District data",
    ])

    with t1:
        c1, c2 = st.columns([2, 1])
        with c1:
            show_img(SPATIAL_DIR / "lisa_map.png")
        with c2:
            st.markdown("""
            <div style='font-size:13px;font-weight:500;color:#1a1a2e;margin-bottom:12px;'>
                LISA cluster types
            </div>
            """, unsafe_allow_html=True)
            for color, label, desc in [
                ("#D85A30", "HH — Hot spots",
                 "High enrolment districts surrounded by other high enrolment districts. "
                 "These are Aadhaar adoption clusters — geographically concentrated success zones."),
                ("#378ADD", "LL — Cold spots",
                 "Low enrolment districts surrounded by other low enrolment districts. "
                 "Regions where Aadhaar adoption is consistently low across a geographic area."),
                ("#EF9F27", "HL — Spatial outliers",
                 "High enrolment district surrounded by low-enrolment neighbours. "
                 "Isolated high performers — possibly district-level campaigns not spreading to neighbours."),
                ("#9FE1CB", "LH — Spatial outliers",
                 "Low enrolment district surrounded by high-enrolment neighbours. "
                 "Districts lagging behind their region — targeted intervention candidates."),
                ("#e5e7eb", "NS — Not significant",
                 "No statistically significant spatial pattern. Enrolment level is not "
                 "meaningfully associated with what neighbours are doing."),
            ]:
                st.markdown(
                    f"<div class='insight-card' style='border-left:3px solid {color};'>"
                    f"<div class='insight-title'>{label}</div>"
                    f"<div class='insight-body'>{desc}</div></div>",
                    unsafe_allow_html=True
                )

    with t2:
        c1, c2 = st.columns([3, 2])
        with c1:
            show_img(SPATIAL_DIR / "moran_scatter.png")
        with c2:
            note_box("Reading the Moran scatter plot",
                "Each dot is one district. The X-axis shows its standardised enrolment "
                "(positive = above average, negative = below average). "
                "The Y-axis shows the <b>spatial lag</b> — the weighted average of its neighbours\' enrolment.<br><br>"
                "The <b>slope of the dashed line</b> is the Global Moran\'s I.<br><br>"
                "Quadrants:<br>"
                "Top-right = HH (hot spots)<br>"
                "Bottom-left = LL (cold spots)<br>"
                "Top-left = LH (low surrounded by high)<br>"
                "Bottom-right = HL (high surrounded by low)<br><br>"
                "A steep positive slope = strong geographic clustering of enrolment.")

    with t3:
        show_img(SPATIAL_DIR / "moran_by_feature.png")
        note_box("Global Moran\'s I across all features",
            "This chart shows whether each Aadhaar feature is spatially clustered "
            "or dispersed across India.<br><br>"
            "<b style='color:#D85A30;'>Orange bars (I > 0)</b> = the feature clusters geographically "
            "— nearby districts tend to have similar values.<br><br>"
            "<b style='color:#378ADD;'>Blue bars (I < 0)</b> = the feature is spatially dispersed "
            "— nearby districts tend to differ.<br><br>"
            "Stars show statistical significance: * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001. "
            "Only starred bars should be interpreted as real spatial patterns.")

    with t4:
        if report_path.exists():
            st.markdown("**Full Moran\'s I report**")
            st.code(report_path.read_text(), language="text")
        else:
            st.warning("moran_report.txt not found.")

    with t5:
        csv_path = SPATIAL_DIR / "spatial_summary.csv"
        if csv_path.exists():
            sdf = pd.read_csv(csv_path)
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                search = st.text_input("Search district", "", key="spatial_search")
            with col_f2:
                lisa_filter = st.selectbox("Filter LISA type",
                    ["All", "HH", "LL", "HL", "LH", "NS"])
            disp = sdf.copy()
            if search:
                disp = disp[disp["district"].str.contains(search, case=False, na=False)]
            if lisa_filter != "All":
                disp = disp[disp["lisa_type"] == lisa_filter]
            disp = disp.sort_values("enrol_total", ascending=False)
            st.dataframe(disp, hide_index=True, use_container_width=True)
            st.download_button("Download spatial_summary.csv",
                               disp.to_csv(index=False),
                               "spatial_summary.csv", "text/csv")
        else:
            st.warning("spatial_summary.csv not found.")






# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — TABLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

elif page == "Table Analysis":

    st.markdown("""
    <div class='section-title'>Table Analysis</div>
    <div class='section-sub'>
        Complete statistics and visualizations for all 3 Aadhaar tables
        across all states and all dates — no district filter
    </div>
    """, unsafe_allow_html=True)

    TD = TABLE_DIR
    if not (TD / "bio_growth_trend.png").exists():
        st.warning("Charts not found. Run table_analysis.py first.")
        st.code("python table_analysis.py --db ../database/aadhar.duckdb")
        st.stop()

    def show(path, wide=True):
        p = TD / path
        if p.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(p), use_container_width=wide)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"{path} not found.")

    def note(body):
        st.markdown(
            f"<div style='background:white;border:1px solid #e5e7eb;"
            f"border-radius:10px;padding:14px 18px;font-size:13px;"
            f"color:#6b7280;line-height:1.7;margin-top:8px;'>{body}</div>",
            unsafe_allow_html=True)

    tab_bio, tab_enrol, tab_demo, tab_combined = st.tabs([
        "Biometric",
        "Enrolment",
        "Demographic",
        "Combined",
    ])

    # ── BIOMETRIC ──────────────────────────────────────────────────────
    with tab_bio:
        st.markdown("<div style='font-size:13px;color:#6b7280;margin-bottom:12px;'>"
                    "Biometric enrolment across all states and all dates — age structure, "
                    "dependency ratios, growth momentum and volatility.</div>",
                    unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**National age group totals**")
            show("bio_age_distribution.png")
        with c2:
            st.markdown("**Top 10 vs Bottom 10 states**")
            show("bio_top_bottom.png")

        st.markdown("**National daily trend — bio total with age breakdown**")
        show("bio_growth_trend.png")
        note("Bars show raw daily totals. Lines show 7-day rolling averages for total, "
             "age 5–17 and age 17+. Vertical lines mark month boundaries.")

        c3,c4 = st.columns(2)
        with c3:
            st.markdown("**State × month dependency heatmap**")
            show("bio_state_heatmap.png")
            note("Each cell = avg biometric dependency ratio for that state in that month. "
                 "Darker = higher ratio of young enrolees relative to adults.")
        with c4:
            st.markdown("**Enrolment volatility by state**")
            show("bio_volatility_map.png")
            note("7-day std of biometric enrolment per state. "
                 "Orange = above-median volatility — spiky, campaign-driven patterns.")

        st.markdown("**Age 5–17 ratio vs dependency — state scatter**")
        show("bio_ratio_scatter.png")
        note("Each dot is one state. Bubble size = total bio enrolment. "
             "Colour = avg daily growth rate (green = growing, red = declining).")

    # ── ENROLMENT ──────────────────────────────────────────────────────
    with tab_enrol:
        st.markdown("<div style='font-size:13px;color:#6b7280;margin-bottom:12px;'>"
                    "Enrolment across all states and all dates — age breakdowns, "
                    "adult vs minor ratios, growth and volatility patterns.</div>",
                    unsafe_allow_html=True)

        c1,c2 = st.columns([1,2])
        with c1:
            st.markdown("**National age group share**")
            show("enrol_age_pie.png")
        with c2:
            st.markdown("**Adult vs minor ratio — all states**")
            show("enrol_adult_minor_bar.png")
            note("Sorted by adult ratio. States at the top have higher adult saturation. "
                 "States at the bottom still have a large share of child enrolments ongoing.")

        st.markdown("**National daily trend — enrolment total by age group**")
        show("enrol_trend.png")
        note("All 4 series smoothed with a 7-day rolling average. "
             "Age 18+ dominates nationally but watch for states where 0–5 spikes seasonally.")

        c3,c4 = st.columns(2)
        with c3:
            st.markdown("**State × month growth rate heatmap**")
            show("enrol_growth_heatmap.png")
            note("Green = positive growth, Red = decline. "
                 "Cells show avg daily % change for each state in each month.")
        with c4:
            st.markdown("**Growth momentum ranking**")
            show("enrol_top_growth.png")
            note("Orange = above-median growth. Purple dashed line = national median. "
                 "States above it had active enrolment drives during the window.")

        st.markdown("**Enrolment volatility by state**")
        show("enrol_volatility.png")
        note("High volatility = irregular enrolment bursts rather than steady flow. "
             "Investigate the top volatile states for campaign-driven patterns.")

    # ── DEMOGRAPHIC ────────────────────────────────────────────────────
    with tab_demo:
        st.markdown("<div style='font-size:13px;color:#6b7280;margin-bottom:12px;'>"
                    "Demographic enrolment across all states and all dates — age ratios, "
                    "dependency patterns and state comparisons.</div>",
                    unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Dependency ratio — distribution + state ranking**")
            show("demo_dependency_dist.png")
            note("Left: distribution of avg dependency ratios across states. "
                 "Right: per-state ranking. Orange = above national median.")
        with c2:
            st.markdown("**Age group split — top 30 states**")
            show("demo_state_comparison.png")
            note("Stacked bar showing age 5–17 vs age 17+ demographic enrolment. "
                 "Sorted by total. Shows which states have younger demographic profiles.")

        st.markdown("**National daily trend — demographic total by age group**")
        show("demo_trend.png")
        note("Smoothed daily totals for demographic enrolment. "
             "Compare the gap between age groups across months to spot demographic shifts.")

        st.markdown("**State × month age 5–17 ratio heatmap**")
        show("demo_age_ratio_heatmap.png")
        note("Darker cells = higher proportion of age 5–17 in demographic enrolment. "
             "Consistent dark rows = structurally young districts in that state.")

    # ── COMBINED ───────────────────────────────────────────────────────
    with tab_combined:
        st.markdown("<div style='font-size:13px;color:#6b7280;margin-bottom:12px;'>"
                    "Cross-table analysis — all 3 tables on the same timeline, "
                    "and a feature correlation matrix across bio × enrolment × demographic.</div>",
                    unsafe_allow_html=True)

        st.markdown("**All 3 tables — national daily totals on the same chart**")
        show("all_tables_trend.png")
        note("Biometric (purple) · Enrolment (green) · Demographic (amber). "
             "All smoothed with a 7-day rolling average. "
             "The gap between tables shows which enrolment type is leading or lagging nationally. "
             "Shaded areas show the full range of daily values.")

        st.markdown("**Feature correlation matrix — Bio × Enrolment × Demographic**")
        show("correlation_heatmap.png")
        note("Lower triangle only (symmetric matrix). Each cell = Pearson r between two state-level features.<br>"
             "<b>Green (r → 1)</b> = strong positive correlation — these features move together.<br>"
             "<b>Red (r → -1)</b> = strong negative correlation — one rises as the other falls.<br>"
             "<b>Near-zero</b> = independent features, no linear relationship.<br>"
             "Use this to identify which engineered features carry the same information "
             "and which ones add independent signal for clustering or STGCN.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — DISTRICT DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════

elif page == "District Deep-Dive":
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    import io

    st.markdown("""
    <div class='section-title'>District Deep-Dive</div>
    <div class='section-sub'>
        Select any district — complete profile with KPIs, time series
        across all 70 dates, and state peer comparison
    </div>
    """, unsafe_allow_html=True)

    prof_path = TABLE_DIR / "district_profiles.csv"
    ts_path   = TABLE_DIR / "district_timeseries.csv"

    if not prof_path.exists():
        st.warning("Run district_analysis.py first.")
        st.code("python table_analysis.py --db ../database/aadhar.duckdb")
        st.stop()

    @st.cache_data
    def load_dd():
        p = pd.read_csv(prof_path)
        t = pd.read_csv(ts_path, parse_dates=["date"])
        return p, t

    prof, ts = load_dd()
    BG = "#f9f8f6"

    cs, cd = st.columns([1, 2])
    with cs:
        sel_state = st.selectbox("State",
            sorted(prof["state"].dropna().unique()), key="dd_s")
    with cd:
        sel_dist = st.selectbox("District",
            sorted(prof[prof["state"]==sel_state]["district"].dropna().unique()), key="dd_d")

    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:.6rem 0 1rem'>",
                unsafe_allow_html=True)

    row = prof[(prof["district"]==sel_dist)&(prof["state"]==sel_state)]
    if row.empty:
        st.warning("No data for this district."); st.stop()
    row = row.iloc[0]

    def val(col, fmt="num"):
        v = row.get(col, None)
        if v is None or (isinstance(v, float) and np.isnan(float(v))): return "—"
        v = float(v)
        if fmt=="num":  return f"{v/1e6:.2f}M" if v>=1e6 else f"{v/1e3:.1f}K" if v>=1e3 else f"{v:.0f}"
        if fmt=="pct":  return f"{v*100:.2f}%"
        if fmt=="dec2": return f"{v:.2f}"
        if fmt=="dec3": return f"{v:.3f}"
        if fmt=="rank": return f"#{int(v)}"
        return f"{v:.2f}"

    tier  = row.get("performance_tier","—")
    tierc = {"Top 10%":"#1D9E75","Top 25%":"#534AB7","Above average":"#BA7517",
             "Below average":"#D85A30","Bottom 25%":"#9ca3af"}.get(tier,"#9ca3af")

    # KPI banner
    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='metric-label'>Enrolment total</div>
            <div class='metric-value'>{val("enrol_total")}</div>
            <div class='metric-note'>Apr–Jul 2025</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Biometric total</div>
            <div class='metric-value'>{val("bio_total")}</div>
            <div class='metric-note'>Apr–Jul 2025</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Demographic total</div>
            <div class='metric-value'>{val("demo_total")}</div>
            <div class='metric-note'>Apr–Jul 2025</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>National rank</div>
            <div class='metric-value'>{val("enrol_total_national_rank","rank")}</div>
            <div class='metric-note'>of {len(prof)} districts</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>State rank</div>
            <div class='metric-value'>{val("enrol_total_state_rank","rank")}</div>
            <div class='metric-note'>in {sel_state[:18]}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Percentile</div>
            <div class='metric-value'>{val("enrol_total_percentile","dec2")}th</div>
            <div class='metric-note' style='color:{tierc};font-weight:500;'>{tier}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Three summary cards
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='insight-card' style='border-left:3px solid #534AB7;'>
            <div class='insight-cluster' style='color:#534AB7;'>Biometric</div>
            <div class='insight-body'>
                <b>Age 5–17:</b> {val("bio_age_5_17_total")}<br>
                <b>Age 17+:</b> {val("bio_age_17_plus_total")}<br>
                <b>Dependency:</b> {val("bio_dependency_ratio","dec3")}<br>
                <b>7-day avg:</b> {val("bio_7day_avg")}<br>
                <b>Volatility:</b> {val("bio_7day_std","dec2")}<br>
                <b>Daily growth:</b> {val("bio_daily_pct_change","dec2")}%<br>
                <b>State rank:</b> {val("bio_rank_in_state","dec2")}
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='insight-card' style='border-left:3px solid #1D9E75;'>
            <div class='insight-cluster' style='color:#1D9E75;'>Enrolment</div>
            <div class='insight-body'>
                <b>Age 0–5:</b> {val("enrol_age_0_5_total")}<br>
                <b>Age 5–17:</b> {val("enrol_age_5_17_total")}<br>
                <b>Age 18+:</b> {val("enrol_age_18_plus_total")}<br>
                <b>Minor ratio:</b> {val("enrol_minor_ratio","pct")}<br>
                <b>Adult ratio:</b> {val("enrol_adult_ratio","pct")}<br>
                <b>Daily growth:</b> {val("enrol_daily_pct_change","dec2")}%<br>
                <b>Volatility:</b> {val("enrol_7day_std","dec2")}
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='insight-card' style='border-left:3px solid #BA7517;'>
            <div class='insight-cluster' style='color:#BA7517;'>Demographic</div>
            <div class='insight-body'>
                <b>Age 5–17:</b> {val("demo_age_5_17_total")}<br>
                <b>Age 17+:</b> {val("demo_age_17_plus_total")}<br>
                <b>Age 5 ratio:</b> {val("demo_age5_ratio","pct")}<br>
                <b>Age 17 ratio:</b> {val("demo_age17_ratio","pct")}<br>
                <b>Dependency:</b> {val("demo_dependency_ratio","dec3")}<br>
                <b>Daily growth:</b> {val("demo_daily_pct_change","dec2")}%<br>
                <b>State rank:</b> {val("demo_rank_in_state","dec2")}
            </div>
        </div>""", unsafe_allow_html=True)

    # Time series
    dist_ts = ts[(ts["district"]==sel_dist)&(ts["state"]==sel_state)].sort_values("date")
    if not dist_ts.empty:
        metric_sel = st.selectbox("Feature to plot", [
            "enrol_total","bio_total","demo_total",
            "enrol_minor_ratio","enrol_adult_ratio","bio_dependency",
            "enrol_pct_change","bio_pct_change",
        ], key="dd_metric")

        fig, axes = plt.subplots(2,1, figsize=(13,6), facecolor=BG,
                                  gridspec_kw={"height_ratios":[3,1],"hspace":0.08})
        ax = axes[0]; ax.set_facecolor("#ffffff")
        vals   = dist_ts[metric_sel].fillna(0)
        smooth = vals.rolling(7, min_periods=1, center=True).mean()
        ax.bar(dist_ts["date"], vals, color="#1D9E75", alpha=0.18, width=0.8)
        ax.plot(dist_ts["date"], smooth, color="#1D9E75", linewidth=2.5)
        for ms in ["2025-09-01","2025-10-01","2025-11-01","2025-12-01"]:
            ax.axvline(pd.Timestamp(ms), color="#e5e7eb", linewidth=1)
            ax.text(pd.Timestamp(ms),
                    smooth.max()*1.01 if smooth.max()>0 else 1,
                    f" {pd.Timestamp(ms).strftime('%b')}", fontsize=8, color="#9ca3af")
        ax.set_title(f"{sel_dist} — {metric_sel}", fontsize=11,
                     fontweight="bold", loc="left", pad=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x,_: f"{x/1e3:.0f}K" if x>=1e3 else f"{x:.3f}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        plt.setp(ax.get_xticklabels(), visible=False)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax.spines[sp].set_color("#e5e7eb")
        ax.grid(True, axis="y", color="#f0f0f0")

        ax2 = axes[1]; ax2.set_facecolor("#ffffff")
        g = dist_ts["enrol_pct_change"].fillna(0)
        ax2.bar(dist_ts["date"], g,
                color=["#1D9E75" if x>=0 else "#D85A30" for x in g],
                alpha=0.75, width=0.8)
        ax2.axhline(0, color="#9ca3af", linewidth=0.8)
        ax2.set_ylabel("Growth %", fontsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        plt.setp(ax2.get_xticklabels(), rotation=28, ha="right", fontsize=7)
        for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax2.spines[sp].set_color("#e5e7eb")
        ax2.grid(True, axis="y", color="#f0f0f0")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(); buf.seek(0)
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(buf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Peer comparison
    with st.expander(f"Compare with all districts in {sel_state}"):
        peers = prof[prof["state"]==sel_state].sort_values("enrol_total",ascending=False).copy()
        pcols = ["district","enrol_total","bio_total","demo_total",
                 "enrol_minor_ratio","enrol_adult_ratio","bio_dependency_ratio",
                 "enrol_daily_pct_change","enrol_total_state_rank"]
        pcols = [c for c in pcols if c in peers.columns]
        pdisp = peers[pcols].copy()
        pdisp[pdisp.select_dtypes("float").columns] = pdisp.select_dtypes("float").round(3)
        def hl(r):
            return ["background:#eef2ff;font-weight:500"
                    if r["district"]==sel_dist else "" for _ in r]
        st.dataframe(pdisp.style.apply(hl, axis=1),
                     hide_index=True, use_container_width=True)
        st.download_button("Download CSV", pdisp.to_csv(index=False),
                           f"{sel_state}_peers.csv","text/csv")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 6 — STGCN RESULTS
# ══════════════════════════════════════════════════════════════════════════

elif page == "STGCN Results":
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import io

    st.markdown("""
    <div class='section-title'>STGCN Results</div>
    <div class='section-sub'>
        Spatio-Temporal Graph Convolutional Network —
        district-level Aadhaar enrolment forecasting across 945 districts
    </div>
    """, unsafe_allow_html=True)

    if not (STGCN_DIR / "metrics.txt").exists():
        st.warning("Run stgcn_train.py first to generate results.")
        st.code("python stgcn_train.py")
        st.stop()

    # ── Parse metrics.txt ──────────────────────────────────────────────
    metrics_txt = (STGCN_DIR / "metrics.txt").read_text()
    rows_parsed = []
    for line in metrics_txt.split("\n"):
        line = line.strip()
        if not line or line.startswith(("=","All","MAE","sMAPE","R2","mMAPE",
                                        "-","MEAN","INTERP","Note","Subst","  R2")):
            continue
        parts = line.split()
        # format: Feature MAE RMSE sMAPE% R2 mMAPE%
        if len(parts) >= 5:
            try:
                fname = parts[0] if "(noisy)" not in line else parts[0]+" (noisy)"
                # find numeric values: MAE RMSE sMAPE R2 [mMAPE]
                nums = []
                for p in parts[1:]:
                    try: nums.append(float(p.strip("%")))
                    except: pass
                if len(nums) >= 4:
                    rows_parsed.append({
                        "Feature": fname,
                        "MAE":   nums[0],
                        "RMSE":  nums[1],
                        "sMAPE": nums[2],
                        "R2":    nums[3],
                    })
            except: pass

    # Extract R2 values for key metrics display
    r2_vals   = [r["R2"] for r in rows_parsed if "(noisy)" not in r["Feature"]]
    mae_vals  = [r["MAE"] for r in rows_parsed]
    mean_r2   = float(np.mean(r2_vals)) if r2_vals else 0.0
    mean_mae  = float(np.mean(mae_vals)) if mae_vals else 0.0
    best_feat = max(rows_parsed, key=lambda x: x["R2"]) if rows_parsed else {}
    worst_feat= min([r for r in rows_parsed if "(noisy)" not in r["Feature"]],
                    key=lambda x: x["R2"]) if rows_parsed else {}

    # ── KPI banner ─────────────────────────────────────────────────────
    r2_label = ("Good" if mean_r2 > 0.5 else
                "Moderate" if mean_r2 > 0.3 else "Building")
    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='metric-label'>Mean R² (ex growth)</div>
            <div class='metric-value'>{mean_r2:.3f}</div>
            <div class='metric-note'>{r2_label} — variance explained</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Mean MAE</div>
            <div class='metric-value'>{mean_mae:.4f}</div>
            <div class='metric-note'>on z-scored [0,1] scale</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Best feature</div>
            <div class='metric-value' style='font-size:1rem;'>{best_feat.get("Feature","—")[:18]}</div>
            <div class='metric-note'>R² = {best_feat.get("R2",0):.3f}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Districts</div>
            <div class='metric-value'>945</div>
            <div class='metric-note'>forecasted simultaneously</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Training sequences</div>
            <div class='metric-value'>37</div>
            <div class='metric-note'>T=70 dates, T_in=6</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Architecture</div>
            <div class='metric-value' style='font-size:1rem;'>STGCN</div>
            <div class='metric-note'>2 ST-Conv blocks · K=4 · C=7</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    def show(path, wide=True):
        p = STGCN_DIR / path
        if p.exists():
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(str(p), use_container_width=wide)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"{path} not found.")

    def note(body):
        st.markdown(
            f"<div style='background:white;border:1px solid #e5e7eb;"
            f"border-radius:10px;padding:14px 18px;font-size:13px;"
            f"color:#6b7280;line-height:1.7;'>{body}</div>",
            unsafe_allow_html=True)

    t1,t2,t3,t4,t5 = st.tabs([
        "R² by feature",
        "Loss curve",
        "Predicted vs actual",
        "District errors",
        "Full metrics",
    ])

    # ── Tab 1: R² bar chart ─────────────────────────────────────────────
    with t1:
        if rows_parsed:
            fig, ax = plt.subplots(figsize=(9,5), facecolor="#f9f8f6")
            ax.set_facecolor("#ffffff")
            feats = [r["Feature"][:22] for r in rows_parsed]
            r2s   = [r["R2"] for r in rows_parsed]
            colors = ["#1D9E75" if r > 0.4 else
                      "#BA7517" if r > 0.2 else
                      "#D85A30" for r in r2s]
            bars = ax.barh(feats, r2s, color=colors, alpha=0.85, height=0.6)
            ax.axvline(0,    color="#9ca3af", linewidth=0.8)
            ax.axvline(0.3,  color="#534AB7", linewidth=1, linestyle="--",
                       alpha=0.6, label="R²=0.3 (moderate)")
            ax.axvline(0.5,  color="#1D9E75", linewidth=1, linestyle="--",
                       alpha=0.6, label="R²=0.5 (good)")
            for bar, v in zip(bars, r2s):
                ax.text(max(v+0.01, 0.01), bar.get_y()+bar.get_height()/2,
                        f"{v:.3f}", va="center", fontsize=9, color="#374151")
            ax.set_xlabel("R² (explained variance)", fontsize=10)
            ax.set_title("STGCN — R² per feature",
                         fontsize=13, fontweight="bold", loc="left", pad=10)
            ax.legend(fontsize=8)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            for sp in ["bottom","left"]: ax.spines[sp].set_color("#e5e7eb")
            ax.grid(True, axis="x", color="#f0f0f0")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150,
                        bbox_inches="tight", facecolor="#f9f8f6")
            plt.close(); buf.seek(0)
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(buf, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            note(
                "<b>Reading R²:</b> 1.0 = perfect prediction, 0.0 = predicting the mean, "
                "negative = worse than the mean.<br><br>"
                "<b style='color:#1D9E75;'>Green</b> = good (R² > 0.4) — "
                "model captures district variation well.<br>"
                "<b style='color:#BA7517;'>Amber</b> = moderate (0.2–0.4).<br>"
                "<b style='color:#D85A30;'>Red</b> = weak — feature is too noisy "
                "or lacks sufficient temporal signal.<br><br>"
                "enrol_growth_pct is expected to be red — daily % change is an "
                "inherently noisy derivative signal with R²≈0.009."
            )

    # ── Tab 2: Loss curve ───────────────────────────────────────────────
    with t2:
        show("loss_curve.png")
        note(
            "Training loss (purple) and validation loss (green) over epochs. "
            "The model uses <b>CosineAnnealingWarmRestarts</b> — LR oscillates "
            "smoothly rather than decaying monotonically, which helps escape "
            "local minima on small datasets.<br><br>"
            "If val loss stops improving while train loss keeps dropping, "
            "the model is overfitting. Early stopping (patience=25) prevents this."
        )

    # ── Tab 3: Predicted vs actual ──────────────────────────────────────
    with t3:
        show("pred_vs_actual.png")
        note(
            "6 randomly sampled districts — predicted enrolment (dashed red) vs "
            "actual enrolment (solid purple) on the test set.<br><br>"
            "Each subplot shows the model's forecast across test time steps. "
            "MAE per district is shown in the title. "
            "Districts with low MAE have consistent enrolment patterns "
            "(clusters 0, 1, 3). "
            "High-MAE districts are typically from cluster 2 (campaign spikes) "
            "or cluster 4 (border/restricted zones)."
        )

    # ── Tab 4: District errors ──────────────────────────────────────────
    with t4:
        csv_path = STGCN_DIR / "per_district_error.csv"
        if csv_path.exists():
            edf = pd.read_csv(csv_path)

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                search_e = st.text_input("Search district", "", key="stgcn_search")
            with col_f2:
                sort_e = st.selectbox("Sort by", ["mae","smape","r2"],
                                      key="stgcn_sort")

            disp_e = edf.copy()
            if search_e:
                disp_e = disp_e[disp_e["district"].str.contains(
                    search_e, case=False, na=False)]
            asc = sort_e == "r2"  # r2: higher is better so sort desc
            disp_e = disp_e.sort_values(sort_e, ascending=asc)
            disp_e[["mae","smape","r2"]] = disp_e[["mae","smape","r2"]].round(4)
            st.dataframe(disp_e, hide_index=True, use_container_width=True)
            st.download_button("Download per_district_error.csv",
                               disp_e.to_csv(index=False),
                               "per_district_error.csv", "text/csv")

            # Quick inline MAE distribution chart
            fig, ax = plt.subplots(figsize=(8, 3), facecolor="#f9f8f6")
            ax.set_facecolor("#ffffff")
            ax.hist(edf["mae"].dropna(), bins=50,
                    color="#534AB7", alpha=0.75, edgecolor="white")
            ax.axvline(edf["mae"].mean(), color="#D85A30", linewidth=1.5,
                       linestyle="--",
                       label=f"Mean MAE = {edf['mae'].mean():.4f}")
            ax.set_xlabel("MAE (enrol_total)", fontsize=9)
            ax.set_title("Per-district MAE distribution",
                         fontsize=10, fontweight="bold", loc="left")
            ax.legend(fontsize=8)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            for sp in ["bottom","left"]: ax.spines[sp].set_color("#e5e7eb")
            ax.grid(True, axis="y", color="#f0f0f0")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150,
                        bbox_inches="tight", facecolor="#f9f8f6")
            plt.close(); buf.seek(0)
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(buf, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            note("Most districts cluster near the mean MAE. "
                 "The long right tail is volatile / border districts. "
                 "Cross-reference with the Clustering page — "
                 "high-error districts are typically Cluster 2 or Cluster 4.")
        else:
            st.warning("per_district_error.csv not found.")

    # ── Tab 5: Full metrics text ────────────────────────────────────────
    with t5:
        st.code(metrics_txt, language="text")
        st.markdown("""
        <div style='background:white;border:1px solid #e5e7eb;
                    border-radius:10px;padding:16px 20px;
                    font-size:13px;color:#6b7280;line-height:1.8;'>
            <b style='color:#1a1a2e;'>Model configuration</b><br><br>
            Architecture: STGCN with 2 Spatio-Temporal Conv blocks<br>
            Graph conv: Chebyshev K=4 (4-hop neighbourhood)<br>
            Temporal conv: kernel Kt=2 with GLU gating<br>
            Input window: T_in=6 time steps<br>
            Prediction: 1 step ahead per district<br>
            Nodes: 945 districts · Features: C=7 per node<br>
            Adjacency: W = 0.5×W_distance + 0.5×W_similarity<br>
            Normalisation: z-score per feature (mean=0, std=1)<br>
            Training: 37 sequences · Validation: 12 · Test: 13<br>
            Optimiser: Adam · Scheduler: CosineAnnealingWarmRestarts<br>
            Loss: MSE · Early stopping: patience=25
        </div>
        """, unsafe_allow_html=True)