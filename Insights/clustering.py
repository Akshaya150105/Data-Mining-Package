"""
clustering.py  —  runs on top of your *_preprocessed tables
"""
from district_mapping import DISTRICT_MAP
import argparse, warnings
from pathlib import Path
import duckdb, geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import folium
from rapidfuzz import fuzz, process
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path("clustering_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MONTHLY_DATES = ['2025-04-01','2025-05-01','2025-06-01','2025-07-01']

BIO_FEATS = [
    'age_5_ratio',
    'age_17_ratio',
    'dependency_ratio',
    'bio_total_7day_std',
    'district_rank_in_state',
    'daily_pct_change',
]

DEMO_FEATS = [
    'demo_age_5_ratio',
    'demo_age_17_ratio',
    'demo_dependency_ratio',
    'demo_total_7day_std',
    'district_rank_in_state',
    'daily_pct_change',
]

ENROL_FEATS = [
    'enrol_minor_ratio',
    'enrol_adult_ratio',
    'enrol_total_7day_std',
    'district_rank_in_state',
    'daily_pct_change',
]

ALL_FEATURES = (
    [f'bio__{c}'   for c in BIO_FEATS]  +
    [f'demo__{c}'  for c in DEMO_FEATS] +
    [f'enrol__{c}' for c in ENROL_FEATS]
)

FUZZY_THRESH = 82
COLORS = ['#1D9E75','#534AB7','#D85A30','#BA7517',
          '#D4537E','#378ADD','#639922','#E24B4A','#5DCAA5','#7F77DD']

# ── Step 1: fetch ──────────────────────────────────────────────────────────
def fetch_features(db_path):
    con  = duckdb.connect(db_path, read_only=True)

    # Auto-detect common dates across all 3 preprocessed tables
    bio_dates   = set(con.execute("SELECT DISTINCT CAST(date AS DATE)::VARCHAR FROM biometric_data_preprocessed").df().iloc[:,0])
    demo_dates  = set(con.execute("SELECT DISTINCT CAST(date AS DATE)::VARCHAR FROM demographic_data_preprocessed").df().iloc[:,0])
    enrol_dates = set(con.execute("SELECT DISTINCT CAST(date AS DATE)::VARCHAR FROM enrolment_data_preprocessed").df().iloc[:,0])

    common = sorted(bio_dates & demo_dates & enrol_dates)

    # Keep only the 4 clean monthly snapshots for clustering
    MONTHLY_DATES = [d for d in common if d.endswith('-01')]
    print(f"  Monthly dates found: {MONTHLY_DATES}")

    dts = ", ".join([f"'{d}'" for d in MONTHLY_DATES])

    def fetch(table, cols, prefix):
        sql = ", ".join([f'AVG("{c}") AS "{prefix}__{c}"' for c in cols])
        df  = con.execute(f"""
            SELECT district, state, {sql}
            FROM {table}
            WHERE CAST(date AS DATE)::VARCHAR IN ({dts})
              AND district IS NOT NULL
            GROUP BY district, state
        """).fetchdf()
        print(f"  [{table}] {len(df)} districts")
        return df

    bio   = fetch("biometric_data_preprocessed",   BIO_FEATS,   "bio")
    demo  = fetch("demographic_data_preprocessed",  DEMO_FEATS,  "demo")
    enrol = fetch("enrolment_data_preprocessed",    ENROL_FEATS, "enrol")
    con.close()

    df = bio.merge(demo,  on=["district","state"], how="outer")
    df = df.merge(enrol, on=["district","state"], how="outer").fillna(0)
    print(f"  Combined: {len(df)} districts × {len(ALL_FEATURES)} features")
    return df

# ── Step 2: scale + PCA ───────────────────────────────────────────────────
def prepare(df):
    X = df[ALL_FEATURES].values.astype(np.float64)
    X = np.where(np.isinf(X)|np.isnan(X), 0, X)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca      = PCA(n_components=2, random_state=42)
    X_pca    = pca.fit_transform(X_scaled)
    v = pca.explained_variance_ratio_*100
    print(f"  PCA: PC1={v[0]:.1f}%  PC2={v[1]:.1f}%  total={v.sum():.1f}%")
    return X_scaled, X_pca

# ── Step 3A: K-Means ──────────────────────────────────────────────────────
def run_kmeans(X_scaled, force_k=None):
    k_range = range(2, 11)
    inertias, silhouettes = [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, lbl))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(list(k_range), inertias, 'o-', color='#534AB7', lw=2, markersize=7)
    axes[0].set(xlabel="K", ylabel="Inertia",
                title="Elbow curve — look for the bend")
    axes[0].grid(True, alpha=0.25)

    best_sil_k = list(k_range)[int(np.argmax(silhouettes))]
    axes[1].plot(list(k_range), silhouettes, 's-', color='#1D9E75', lw=2, markersize=7)
    axes[1].axvline(best_sil_k, color='#D85A30', lw=1.5, ls='--',
                    label=f"Best K={best_sil_k}")
    axes[1].set(xlabel="K", ylabel="Silhouette score (higher = better)",
                title="Silhouette — pick the peak")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.25)
    plt.suptitle("Choosing K for K-Means", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"elbow_silhouette.png", dpi=150, bbox_inches='tight')
    plt.close()

    best_k = force_k or best_sil_k
    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)
    print(f"  K-Means K={best_k} | silhouette={max(silhouettes):.3f}")
    for c in range(best_k):
        print(f"    Cluster {c}: {(labels==c).sum()} districts")
    return labels, best_k

# ── Step 3B: DBSCAN ───────────────────────────────────────────────────────
def run_dbscan(X_pca, eps=0.8, min_samples=5):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_pca)
    n  = len(set(labels)) - (1 if -1 in labels else 0)
    nn = (labels==-1).sum()
    print(f"  DBSCAN: {n} clusters, {nn} noise districts")
    for c in sorted(set(labels)):
        print(f"    {'noise' if c==-1 else f'Cluster {c}'}: {(labels==c).sum()}")
    return labels, n

# ── Step 4: PCA scatter ───────────────────────────────────────────────────
def pca_scatter(X_pca, labels, title, fname):
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, c in enumerate(sorted(set(labels))):
        mask  = labels == c
        color = '#BBBBBB' if c==-1 else COLORS[i % len(COLORS)]
        lbl   = 'Noise' if c==-1 else f'Cluster {c}'
        ax.scatter(X_pca[mask,0], X_pca[mask,1],
                   c=color, label=lbl, s=28, alpha=0.75, linewidths=0)
    ax.set(xlabel="PCA component 1", ylabel="PCA component 2", title=title)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PCA scatter → {fname}")

# ── Step 5: Cluster profiles ──────────────────────────────────────────────
def cluster_profiles(df, label_col, fname):
    key = ['bio__age_5_ratio','bio__dependency_ratio','bio__daily_pct_change',
           'demo__demo_age_5_ratio','enrol__enrol_minor_ratio',
           'enrol__enrol_adult_ratio','enrol__daily_pct_change']
    lbls = ['Bio 5-17 ratio','Bio dependency','Bio growth%',
            'Demo 5-17 ratio','Minor ratio',
            'Adult ratio','Enrol growth%']

    profile = df[df[label_col]>=0].groupby(label_col)[key].mean()
    x, width = np.arange(len(key)), 0.75/len(profile)
    fig, ax  = plt.subplots(figsize=(14, 5))
    for i, (cid, row) in enumerate(profile.iterrows()):
        ax.bar(x + i*width, row.values, width,
               label=f'Cluster {cid}',
               color=COLORS[i % len(COLORS)], alpha=0.85)
    ax.set_xticks(x + width*(len(profile)-1)/2)
    ax.set_xticklabels(lbls, rotation=30, ha='right', fontsize=9)
    ax.set(ylabel="Mean value", title="Cluster profiles — key feature averages")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Cluster profiles → {fname}")

def save_summary(df):
    out = ['district','state','kmeans_cluster','dbscan_cluster'] + ALL_FEATURES
    df.sort_values(['kmeans_cluster','state','district'])[out].to_csv(
        OUTPUT_DIR/"cluster_summary.csv", index=False)
    print("  K-Means district counts:")
    print(df.groupby('kmeans_cluster')['district'].count().to_string())

# ── Step 6: Choropleth ────────────────────────────────────────────────────
def norm(s): return " ".join(str(s).lower().split())


def match_shp(shp_path, df):
    gdf = gpd.read_file(shp_path)
    dc  = next((c for c in ["DISTRICT","district","District","DISTNAME"]
                if c in gdf.columns), None)
    if not dc:
        raise ValueError(f"No district col in shapefile.")
    gdf[dc] = gdf[dc].str.strip().str.title()
    shp_districts = gdf[dc].dropna().unique().tolist()
    shp_norm = {norm(s): s for s in shp_districts}

    matched = []
    for d in df['district']:
        # Step 1: apply manual mapping first
        mapped = DISTRICT_MAP.get(d, d)

        # Step 2: fuzzy match the (possibly mapped) name
        res = process.extractOne(norm(mapped), list(shp_norm.keys()), scorer=fuzz.WRatio)
        matched.append(shp_norm[res[0]] if res and res[1] >= 82 else None)

    df2 = df.copy()
    df2['shp_district'] = matched
    unresolved = df2[df2['shp_district'].isna()]['district'].tolist()
    print(f"  Matched: {df2['shp_district'].notna().sum()}/{len(df)} districts")
    if unresolved:
        print(f"  Still unresolved ({len(unresolved)}): {unresolved[:10]}...")

    dissolved = gdf.dissolve(by=dc).reset_index()
    merged    = dissolved.merge(
        df2[['shp_district','kmeans_cluster','dbscan_cluster']],
        left_on=dc, right_on='shp_district', how='left'
    )
    return merged, dc

def static_map(gdf, col, title, fname, n_colors):
    fig, ax = plt.subplots(figsize=(14, 14))
    gdf[gdf[col].isna()].plot(ax=ax, color='#ECECEC',
                               linewidth=0.3, edgecolor='white')
    matched = gdf[gdf[col].notna()].copy()
    matched[col] = matched[col].astype(int)
    cmap = plt.matplotlib.colors.ListedColormap(COLORS[:n_colors])
    matched.plot(ax=ax, column=col, cmap=cmap,
                 linewidth=0.3, edgecolor='white', legend=False)
    patches = [
        mpatches.Patch(
            color='#BBBBBB' if c==-1 else COLORS[c % len(COLORS)],
            label='Noise' if c==-1 else f'Cluster {c}'
        ) for c in sorted(matched[col].unique())
    ]
    ax.legend(handles=patches, loc='lower left', fontsize=10,
              title="Cluster", title_fontsize=11, framealpha=0.92)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/fname, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Static map → {fname}")

def folium_map(gdf, col, title, fname, n_colors, dc):
    wgs = gdf.to_crs("EPSG:4326")
    wgs = wgs[wgs[col].notna()].copy()
    wgs[col] = wgs[col].astype(int)
    m = folium.Map(location=[22, 80], zoom_start=5, tiles='CartoDB positron')

    def style(feat):
        c = feat['properties'].get(col)
        fill = '#BBBBBB' if (c is None or c==-1) else COLORS[int(c)%len(COLORS)]
        return {'fillColor':fill,'color':'white','weight':0.5,'fillOpacity':0.78}

    folium.GeoJson(wgs.__geo_interface__, style_function=style,
        tooltip=folium.features.GeoJsonTooltip(
            fields=[dc, col], aliases=['District:','Cluster:'],localize=True)
    ).add_to(m)

    legend = (f'<div style="position:absolute;bottom:30px;left:30px;z-index:1000;'
              f'background:white;padding:12px 16px;border-radius:8px;'
              f'border:1px solid #ccc;font-family:sans-serif;font-size:13px;">'
              f'<b>{title}</b><br>')
    for i in sorted(wgs[col].unique()):
        c   = '#BBBBBB' if i==-1 else COLORS[int(i)%len(COLORS)]
        lbl = 'Noise' if i==-1 else f'Cluster {i}'
        legend += (f'<div style="margin:4px 0"><span style="display:inline-block;'
                   f'width:14px;height:14px;background:{c};border-radius:2px;'
                   f'margin-right:6px;vertical-align:middle"></span>{lbl}</div>')
    legend += '</div>'
    m.get_root().html.add_child(folium.Element(legend))
    m.save(OUTPUT_DIR/fname)
    print(f"  Interactive map → {fname}")

# ── MAIN ───────────────────────────────────────────────────────────────────
def main(db_path, shp_path, force_k=None, dbscan_eps=0.8, dbscan_min=5):
    print("="*60)
    print("AADHAAR DISTRICT CLUSTERING")
    print(f"DB : {db_path}")
    print(f"Features: {len(ALL_FEATURES)} (from *_preprocessed tables)")
    print("="*60)

    print("\n── 1. Fetch features ──"); df          = fetch_features(db_path)
    print("\n── 2. Scale + PCA ──");   X_s, X_pca  = prepare(df)
    print("\n── 3A. K-Means ──");      km, best_k   = run_kmeans(X_s, force_k)
    print("\n── 3B. DBSCAN ──");       db, n_db     = run_dbscan(X_pca, dbscan_eps, dbscan_min)

    df['kmeans_cluster'] = km
    df['dbscan_cluster']  = db

    print("\n── 4. PCA scatter ──")
    pca_scatter(X_pca, km, f"K-Means (K={best_k})", "pca_kmeans.png")
    pca_scatter(X_pca, db, "DBSCAN", "pca_dbscan.png")

    print("\n── 5. Profiles + summary ──")
    cluster_profiles(df, 'kmeans_cluster', 'cluster_profiles_kmeans.png')
    save_summary(df)

    print("\n── 6. Choropleth maps ──")
    gdf_m, dc = match_shp(shp_path, df)
    static_map(gdf_m,'kmeans_cluster',f"Aadhaar clusters — K-Means (K={best_k})",
               "kmeans_choropleth.png", best_k)
    folium_map(gdf_m,'kmeans_cluster',f"K-Means K={best_k}",
               "kmeans_choropleth.html", best_k, dc)
    static_map(gdf_m,'dbscan_cluster',"Aadhaar clusters — DBSCAN",
               "dbscan_choropleth.png", max(n_db,1))
    folium_map(gdf_m,'dbscan_cluster',"DBSCAN",
               "dbscan_choropleth.html", max(n_db,1), dc)

    print("\n"+"="*60)
    print("DONE — clustering_output/")
    print("  elbow_silhouette.png      look at this FIRST, pick K")
    print("  pca_kmeans.png            are clusters well separated?")
    print("  cluster_profiles_kmeans.png  what each cluster means")
    print("  cluster_summary.csv       district → cluster table")
    print("  kmeans_choropleth.html    open in browser, hover districts")
    print("  kmeans_choropleth.png     put this in your report")
    print("  dbscan_choropleth.html    grey = outlier districts")
    print("="*60)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db",  required=True)
    p.add_argument("--shp", required=True)
    p.add_argument("--k",   type=int,   default=None)
    p.add_argument("--eps", type=float, default=0.8)
    p.add_argument("--min", type=int,   default=5)
    a = p.parse_args()
    main(a.db, a.shp, force_k=a.k, dbscan_eps=a.eps, dbscan_min=a.min)