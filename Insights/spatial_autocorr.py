"""
spatial_autocorr.py
====================
Computes Global and Local Moran's I for Aadhaar enrolment across Indian districts.
Directly uses the W_combined adjacency matrix built earlier.

What Moran's I tells you:
  Global I > 0  → high-enrolment districts cluster near other high-enrolment districts
  Global I < 0  → high-enrolment districts are dispersed (checkerboard pattern)
  Global I ≈ 0  → spatially random

Local Moran's I (LISA) classifies each district as:
  HH (Hot spot)   → high enrolment surrounded by high enrolment
  LL (Cold spot)  → low enrolment surrounded by low enrolment
  HL (Outlier)    → high enrolment surrounded by low (spatial outlier)
  LH (Outlier)    → low enrolment surrounded by high (spatial outlier)
  NS              → not significant

Outputs (in spatial_output/):
    moran_scatter.png         Moran scatter plot (enrolment vs spatial lag)
    lisa_map.png              India map coloured by HH/LL/HL/LH/NS
    moran_by_feature.png      Global Moran's I bar chart for all 7 features
    spatial_summary.csv       District-level LISA results
    moran_report.txt          Global I values + p-values for all features

Usage:
    python spatial_autocorr.py --db ../database/aadhar.duckdb
                               --shp Adjacency_marix/2011_Dist.shp
                               --w   adjacency_output/W_combined.csv
"""

import argparse, warnings
from pathlib import Path
import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path("spatial_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MONTHLY_DATES = ['2025-04-01','2025-05-01','2025-06-01','2025-07-01']
BG    = "#f9f8f6"
WHITE = "#ffffff"
FUZZY_THRESH = 82

# LISA colour palette
LISA_COLORS = {
    "HH": "#D85A30",   # hot spot — coral/red
    "LL": "#378ADD",   # cold spot — blue
    "HL": "#EF9F27",   # high surrounded by low — amber
    "LH": "#9FE1CB",   # low surrounded by high — light teal
    "NS": "#e5e7eb",   # not significant — light grey
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": WHITE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#f0f0f0",
        "grid.linewidth": 0.8, "font.family": "DejaVu Sans", "font.size": 11,
    })


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load W and align districts
# ══════════════════════════════════════════════════════════════════════════

def load_weight_matrix(w_path: str) -> tuple:
    """
    Loads W_combined.csv. Returns (W numpy array, ordered district list).
    Row-normalises W so each row sums to 1 (standard for Moran's I).
    """
    W_df = pd.read_csv(w_path, index_col=0)
    districts = W_df.index.tolist()
    W = W_df.values.astype(np.float64)

    # Row-normalise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # avoid divide-by-zero for isolated districts
    W_norm = W / row_sums

    print(f"  W loaded: {W_norm.shape[0]} districts")
    print(f"  Non-zero edges: {(W > 0).sum() // 2}")
    print(f"  Sparsity: {100*(W == 0).mean():.1f}%")
    return W_norm, districts


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Fetch feature vector from DB, align to W district order
# ══════════════════════════════════════════════════════════════════════════

def fetch_features(db_path: str, w_districts: list) -> pd.DataFrame:
    """
    Pulls district-level aggregated features from preprocessed tables.
    Aligns rows to match the exact order of districts in W.
    """
    con = duckdb.connect(db_path, read_only=True)
    dts = ", ".join([f"'{d}'" for d in MONTHLY_DATES])

    enrol = con.execute(f"""
        SELECT district,
            SUM(enrol_total)          AS enrol_total,
            AVG(enrol_minor_ratio)    AS minor_ratio,
            AVG(enrol_adult_ratio)    AS adult_ratio,
            AVG(daily_pct_change)     AS growth_pct
        FROM enrolment_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts}) AND district IS NOT NULL
        GROUP BY district
    """).fetchdf()

    bio = con.execute(f"""
        SELECT district,
            SUM(bio_total)            AS bio_total,
            AVG(dependency_ratio)     AS dependency_ratio,
            AVG(age_5_ratio)          AS age_5_ratio
        FROM biometric_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts}) AND district IS NOT NULL
        GROUP BY district
    """).fetchdf()

    con.close()

    df = enrol.merge(bio, on="district", how="outer").fillna(0)

    # Fuzzy-match DB district names to W district names (which are shapefile names)
    shp_norm = {" ".join(s.lower().split()): s for s in w_districts}

    def match(d):
        res = process.extractOne(
            " ".join(d.lower().split()), list(shp_norm.keys()), scorer=fuzz.WRatio
        )
        return shp_norm[res[0]] if res and res[1] >= FUZZY_THRESH else None

    df["w_district"] = df["district"].apply(match)
    df = df.dropna(subset=["w_district"])
    df = df.drop_duplicates(subset=["w_district"])
    df = df.set_index("w_district")

    # Reindex to exact W order, fill missing with 0
    df = df.reindex(w_districts).fillna(0)

    print(f"  Features aligned: {df.notna().all(axis=1).sum()} / {len(w_districts)} districts matched")
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Global Moran's I (with permutation test)
# ══════════════════════════════════════════════════════════════════════════

def global_morans_i(x: np.ndarray, W: np.ndarray,
                    n_perms: int = 999) -> tuple:
    """
    Computes Global Moran's I and pseudo p-value via permutation test.
    Returns (I, E[I], z_score, p_value, spatial_lag)
    """
    n = len(x)
    z = (x - x.mean()) / (x.std() + 1e-10)   # standardise

    # Spatial lag: weighted average of neighbours
    spatial_lag = W @ z

    # Moran's I = z' W z / z' z
    I = float(z @ spatial_lag) / float(z @ z)

    # Expected value under randomisation
    E_I = -1.0 / (n - 1)

    # Permutation test
    perm_I = np.zeros(n_perms)
    for i in range(n_perms):
        z_perm = np.random.permutation(z)
        perm_I[i] = float(z_perm @ (W @ z_perm)) / float(z_perm @ z_perm)

    z_score = (I - perm_I.mean()) / (perm_I.std() + 1e-10)
    p_value = (np.abs(perm_I) >= np.abs(I)).mean()   # two-tailed

    return I, E_I, z_score, p_value, spatial_lag, z


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Local Moran's I (LISA)
# ══════════════════════════════════════════════════════════════════════════

def local_morans_i(z: np.ndarray, W: np.ndarray,
                   spatial_lag: np.ndarray,
                   p_threshold: float = 0.05,
                   n_perms: int = 499) -> pd.Series:
    """
    Computes Local Moran's I for each district and classifies as HH/LL/HL/LH/NS.
    """
    n = len(z)
    local_I = z * spatial_lag   # local association

    # Permutation-based significance
    perm_local = np.zeros((n_perms, n))
    for i in range(n_perms):
        z_perm = np.random.permutation(z)
        perm_local[i] = z_perm * (W @ z_perm)

    # p-value: proportion of permutations with |local_I_perm| >= |local_I|
    p_vals = (np.abs(perm_local) >= np.abs(local_I)).mean(axis=0)

    # Classify
    labels = []
    for i in range(n):
        if p_vals[i] > p_threshold:
            labels.append("NS")
        elif z[i] > 0 and spatial_lag[i] > 0:
            labels.append("HH")
        elif z[i] < 0 and spatial_lag[i] < 0:
            labels.append("LL")
        elif z[i] > 0 and spatial_lag[i] < 0:
            labels.append("HL")
        else:
            labels.append("LH")

    print(f"  LISA classification:")
    for lbl in ["HH","LL","HL","LH","NS"]:
        cnt = labels.count(lbl)
        print(f"    {lbl}: {cnt} districts")

    return pd.Series(labels, name="lisa"), p_vals


# ══════════════════════════════════════════════════════════════════════════
# CHART 1 — Moran scatter plot
# ══════════════════════════════════════════════════════════════════════════

def plot_moran_scatter(z, spatial_lag, I, p_value,
                       lisa_labels, districts, feature_name):
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    ax.set_facecolor(WHITE)

    colors = [LISA_COLORS[l] for l in lisa_labels]
    ax.scatter(z, spatial_lag, c=colors, s=22, alpha=0.7,
               linewidths=0.3, edgecolors="white", zorder=3)

    # Regression line
    m, b = np.polyfit(z, spatial_lag, 1)
    xr = np.linspace(z.min(), z.max(), 100)
    ax.plot(xr, m * xr + b, color="#1a1a2e", linewidth=1.8,
            linestyle="--", alpha=0.8, zorder=4)

    # Quadrant lines
    ax.axhline(0, color="#d1d5db", linewidth=0.8)
    ax.axvline(0, color="#d1d5db", linewidth=0.8)

    # Quadrant labels
    for txt, xy in [("HH", (0.75, 0.92)), ("LL", (0.05, 0.08)),
                    ("HL", (0.75, 0.08)), ("LH", (0.05, 0.92))]:
        ax.text(*xy, txt, transform=ax.transAxes,
                fontsize=11, fontweight="bold",
                color=LISA_COLORS.get(txt, "#9ca3af"), alpha=0.6)

    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    ax.set_title(
        f"Moran scatter plot — {feature_name}\n"
        f"Global I = {I:.4f}   p = {p_value:.4f} {sig}",
        fontsize=13, fontweight="bold", pad=12, loc="left"
    )
    ax.set_xlabel("Standardised enrolment (z-score)", fontsize=10)
    ax.set_ylabel("Spatial lag (neighbour average)", fontsize=10)

    # Legend
    patches = [mpatches.Patch(color=LISA_COLORS[l], label=l)
               for l in ["HH","LL","HL","LH","NS"]]
    ax.legend(handles=patches, loc="lower right", fontsize=9,
              title="LISA type", title_fontsize=9, framealpha=0.9)

    ax.spines["bottom"].set_color("#e5e7eb")
    ax.spines["left"].set_color("#e5e7eb")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "moran_scatter.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → moran_scatter.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 2 — LISA map
# ══════════════════════════════════════════════════════════════════════════

def plot_lisa_map(lisa_labels, districts, shp_path):
    gdf = gpd.read_file(shp_path)
    dc  = next((c for c in ["DISTRICT","district","District","DISTNAME"]
                if c in gdf.columns), None)
    if not dc:
        print("  Could not find district column in shapefile — skipping LISA map")
        return

    gdf[dc] = gdf[dc].str.strip().str.title()
    shp_norm = {" ".join(s.lower().split()): s for s in gdf[dc].dropna().unique()}

    result_df = pd.DataFrame({"w_district": districts, "lisa": lisa_labels})

    def match_shp(d):
        res = process.extractOne(
            " ".join(d.lower().split()), list(shp_norm.keys()), scorer=fuzz.WRatio
        )
        return shp_norm[res[0]] if res and res[1] >= FUZZY_THRESH else None

    result_df["shp_name"] = result_df["w_district"].apply(match_shp)
    result_df = result_df.dropna(subset=["shp_name"])

    gdf_d = gdf.dissolve(by=dc).reset_index()
    gdf_d = gdf_d.merge(result_df[["shp_name","lisa"]],
                        left_on=dc, right_on="shp_name", how="left")

    fig, ax = plt.subplots(figsize=(13, 14), facecolor=BG)
    # Unmatched / NS districts
    gdf_d[gdf_d["lisa"].isna()].plot(
        ax=ax, color="#f3f4f6", linewidth=0.3, edgecolor="white")
    # LISA districts
    for label, color in LISA_COLORS.items():
        subset = gdf_d[gdf_d["lisa"] == label]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, linewidth=0.3, edgecolor="white")

    # Legend
    patches = [
        mpatches.Patch(color=LISA_COLORS["HH"], label="HH — Hot spot (high-high)"),
        mpatches.Patch(color=LISA_COLORS["LL"], label="LL — Cold spot (low-low)"),
        mpatches.Patch(color=LISA_COLORS["HL"], label="HL — Spatial outlier (high-low)"),
        mpatches.Patch(color=LISA_COLORS["LH"], label="LH — Spatial outlier (low-high)"),
        mpatches.Patch(color=LISA_COLORS["NS"], label="NS — Not significant"),
        mpatches.Patch(color="#f3f4f6",         label="Unmatched"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=10,
              title="LISA cluster type", title_fontsize=10, framealpha=0.92)
    ax.set_title("Local Moran's I (LISA) — Aadhaar enrolment spatial clusters",
                 fontsize=15, fontweight="bold", pad=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lisa_map.png",
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → lisa_map.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 3 — Global Moran's I for all features
# ══════════════════════════════════════════════════════════════════════════

def plot_moran_by_feature(feat_df: pd.DataFrame, W: np.ndarray):
    features = {
        "enrol_total":       "Total enrolment",
        "minor_ratio":       "Minor ratio (0-17)",
        "adult_ratio":       "Adult ratio (18+)",
        "growth_pct":        "Daily growth %",
        "bio_total":         "Biometric total",
        "dependency_ratio":  "Dependency ratio",
        "age_5_ratio":       "Bio age 5-17 ratio",
    }

    results = []
    for col, label in features.items():
        if col not in feat_df.columns:
            continue
        x = feat_df[col].values.astype(np.float64)
        I, E_I, z_score, p_value, _, _ = global_morans_i(x, W, n_perms=499)
        results.append({"feature": label, "I": I, "p": p_value, "z": z_score})
        print(f"    {label:30s}  I={I:.4f}  p={p_value:.4f}")

    rdf = pd.DataFrame(results).sort_values("I", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    ax.set_facecolor(WHITE)

    colors = ["#D85A30" if I > 0 else "#378ADD" for I in rdf["I"]]
    sig_marker = ["***" if p < 0.001 else "**" if p < 0.01
                  else "*" if p < 0.05 else "ns" for p in rdf["p"]]

    bars = ax.barh(rdf["feature"], rdf["I"],
                   color=colors, alpha=0.85, height=0.6)

    # Significance markers
    for bar, sig in zip(bars, sig_marker):
        w = bar.get_width()
        x_pos = w + 0.003 if w >= 0 else w - 0.003
        ha = "left" if w >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                sig, va="center", ha=ha, fontsize=10,
                color="#374151", fontweight="bold")

    ax.axvline(0, color="#9ca3af", linewidth=0.8)
    ax.set_xlabel("Global Moran's I", fontsize=10)
    ax.set_title(
        "Global Moran's I by feature\n"
        "I > 0 = spatially clustered   I < 0 = dispersed   * p<0.05  ** p<0.01  *** p<0.001",
        fontsize=12, fontweight="bold", pad=12, loc="left"
    )
    ax.spines["bottom"].set_color("#e5e7eb")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "moran_by_feature.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → moran_by_feature.png")
    return rdf


# ══════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════

def save_summary(feat_df, districts, lisa_labels, p_vals,
                 global_I, global_p):
    out = pd.DataFrame({
        "district":    districts,
        "lisa_type":   lisa_labels,
        "lisa_p":      p_vals.round(4),
        "enrol_total": feat_df["enrol_total"].values,
        "bio_total":   feat_df["bio_total"].values,
    })
    out.to_csv(OUTPUT_DIR / "spatial_summary.csv", index=False)
    print(f"  Saved → spatial_summary.csv")

    report_lines = [
        "=" * 55,
        "SPATIAL AUTOCORRELATION REPORT — AADHAAR ENROLMENT",
        "=" * 55,
        "",
        f"Global Moran's I (enrol_total) : {global_I:.6f}",
        f"P-value (permutation test)      : {global_p:.6f}",
        f"Interpretation                  : {'CLUSTERED (I > 0)' if global_I > 0 else 'DISPERSED (I < 0)'}",
        "",
        "LISA classification (enrol_total):",
        f"  HH (hot spots)   : {lisa_labels.count('HH')} districts",
        f"  LL (cold spots)  : {lisa_labels.count('LL')} districts",
        f"  HL (outliers)    : {lisa_labels.count('HL')} districts",
        f"  LH (outliers)    : {lisa_labels.count('LH')} districts",
        f"  NS (not sig.)    : {lisa_labels.count('NS')} districts",
        "",
        "Top HH districts (Aadhaar hot spots):",
    ]
    hh = out[out["lisa_type"] == "HH"].nlargest(10, "enrol_total")
    for _, row in hh.iterrows():
        report_lines.append(f"  {row['district']}")

    report_lines += [
        "",
        "Top LL districts (Aadhaar cold spots):",
    ]
    ll = out[out["lisa_type"] == "LL"].nsmallest(10, "enrol_total")
    for _, row in ll.iterrows():
        report_lines.append(f"  {row['district']}")

    (OUTPUT_DIR / "moran_report.txt").write_text("\n".join(report_lines))
    print("  Saved → moran_report.txt")
    print("\n" + "\n".join(report_lines[:20]))


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main(db_path, shp_path, w_path):
    print("=" * 60)
    print("SPATIAL AUTOCORRELATION (MORAN'S I)")
    print("=" * 60)
    np.random.seed(42)
    setup_style()

    print("\n── Step 1: Load adjacency matrix ──")
    W, districts = load_weight_matrix(w_path)

    print("\n── Step 2: Fetch features + align to W ──")
    feat_df = fetch_features(db_path, districts)

    # Primary feature: enrol_total
    x = feat_df["enrol_total"].values.astype(np.float64)

    print("\n── Step 3: Global Moran's I (enrol_total) ──")
    I, E_I, z_score, p_value, spatial_lag, z = global_morans_i(x, W)
    print(f"  Global I = {I:.6f}  (E[I] = {E_I:.6f})")
    print(f"  Z-score  = {z_score:.4f}")
    print(f"  P-value  = {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")

    print("\n── Step 4: Local Moran's I (LISA) ──")
    lisa, p_vals = local_morans_i(z, W, spatial_lag)
    lisa_list = lisa.tolist()

    print("\n── Step 5: Global Moran's I for all features ──")
    moran_df = plot_moran_by_feature(feat_df, W)

    print("\n── Step 6: Charts ──")
    plot_moran_scatter(z, spatial_lag, I, p_value, lisa_list, districts, "enrol_total")
    plot_lisa_map(lisa_list, districts, shp_path)

    print("\n── Step 7: Save summary ──")
    save_summary(feat_df, districts, lisa_list, p_vals, I, p_value)

    print("\n" + "=" * 60)
    print("DONE — spatial_output/")
    print("  moran_scatter.png     Moran scatter + LISA colour coding")
    print("  lisa_map.png          India map: HH/LL/HL/LH/NS districts")
    print("  moran_by_feature.png  Global I for all 7 features")
    print("  spatial_summary.csv   District-level LISA results")
    print("  moran_report.txt      Global I values + top HH/LL districts")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db",  required=True, help="Path to aadhar.duckdb")
    p.add_argument("--shp", required=True, help="Path to 2011_Dist.shp")
    p.add_argument("--w",   required=True, help="Path to W_combined.csv")
    a = p.parse_args()
    main(a.db, a.shp, a.w)