"""
build_district_graph.py
=======================

Builds a district graph for STGCN using a COMBINED graph:
  1) Distance-based weights from district centroid distance
  2) Similarity-based weights from simple district profiles
  3) W_combined = alpha * W_distance + (1 - alpha) * W_similarity

This updated version is designed for the new workflow:
  - separate biometric model
  - separate enrolment model
  - weekly tensors built later
  - one shared district graph for both

Outputs (in graph_output/):
    W_distance.csv
    W_similarity.csv
    W_combined.csv
    L_normalised_laplacian.csv
    district_order.csv
    district_mapping.csv
    summary.txt
    *_heatmap.png

Usage:
    python build_district_graph.py \
        --shp Adjacency_marix/2011_Dist.shp \
        --db database/aadhar.duckdb \
        --alpha 0.5 --sigma2 0.1 --epsilon 0.1

Dependencies:
    pip install geopandas duckdb numpy pandas scipy matplotlib seaborn rapidfuzz
"""

import argparse
import sys
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rapidfuzz import fuzz, process
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

TABLES = [
    "biometric_data_preprocessed",
    "enrolment_data_preprocessed",
]

FUZZY_THRESH = 85

OUTPUT_DIR = Path("graph_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalise_name(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def load_shapefile(shp_path: str):
    gdf = gpd.read_file(shp_path)

    dist_col = next(
        (c for c in ["DISTRICT", "district", "District", "DISTNAME", "dtname"]
         if c in gdf.columns),
        None
    )
    if dist_col is None:
        raise ValueError(f"No district column found in shapefile. Columns: {list(gdf.columns)}")

    gdf[dist_col] = gdf[dist_col].astype(str).str.strip().str.title()
    return gdf, dist_col


def fetch_db_districts(con):
    districts = set()
    for tbl in TABLES:
        rows = con.execute(
            f"""
            SELECT DISTINCT district
            FROM {tbl}
            WHERE district IS NOT NULL
            """
        ).fetchall()
        districts.update(r[0] for r in rows if r[0] is not None)
    return sorted(districts)


def fuzzy_match_districts(db_districts, shp_districts, threshold=FUZZY_THRESH):
    shp_norm_to_raw = {normalise_name(s): s for s in shp_districts}

    mapping = {}
    unmatched = []

    shp_keys = list(shp_norm_to_raw.keys())

    for db_name in db_districts:
        result = process.extractOne(
            normalise_name(db_name),
            shp_keys,
            scorer=fuzz.WRatio
        )
        if result and result[1] >= threshold:
            mapping[db_name] = shp_norm_to_raw[result[0]]
        else:
            unmatched.append(db_name)

    return mapping, unmatched


# ---------------------------------------------------------------------
# DISTANCE GRAPH
# ---------------------------------------------------------------------

def build_distance_matrix(gdf, dist_col, districts, sigma2=0.1, epsilon=0.1):
    """
    Build W_distance using Gaussian kernel on centroid distances.

    W_ij = exp(-(d_ij^2)/sigma2), then threshold by epsilon.
    """
    sub = gdf[gdf[dist_col].isin(districts)].copy()

    # dissolve to one geometry per district, then reindex in exact node order
    sub = sub.dissolve(by=dist_col).reset_index()
    sub = sub.set_index(dist_col).reindex(districts)

    if sub.geometry.isna().any():
        missing = sub[sub.geometry.isna()].index.tolist()
        raise ValueError(f"Some districts missing geometry after reindex: {missing[:10]}")

    # project if currently lat/lon
    if sub.crs and sub.crs.is_geographic:
        sub = sub.to_crs("EPSG:32643")

    centroids = sub.geometry.centroid
    coords = np.array([[pt.x, pt.y] for pt in centroids])

    D = cdist(coords, coords, metric="euclidean")
    d_max = D.max()
    D_norm = D / d_max if d_max > 0 else D

    W = np.exp(-(D_norm ** 2) / sigma2)
    W[W < epsilon] = 0.0
    np.fill_diagonal(W, 0.0)

    return W.astype(np.float32)


# ---------------------------------------------------------------------
# SIMILARITY GRAPH
# ---------------------------------------------------------------------

def build_similarity_profile(con, db_to_shp, districts):
    """
    Build a small, stable district profile for similarity graph construction.

    We intentionally use only a few meaningful aggregated columns, not lots of
    engineered change/rank/rolling columns.

    Output shape: [N, F]
    """
    # Biometric district profile
    bio_sql = """
    SELECT
        district,
        AVG(bio_total) AS bio_total_mean,
        STDDEV_SAMP(bio_total) AS bio_total_std,
        AVG(age_5_ratio) AS bio_age5_ratio_mean,
        AVG(age_17_ratio) AS bio_age17_ratio_mean,
        AVG(dependency_ratio) AS bio_dependency_mean
    FROM biometric_data_preprocessed
    WHERE district IS NOT NULL
    GROUP BY district
    """

    # Enrolment district profile
    enr_sql = """
    SELECT
        district,
        AVG(enrol_total) AS enrol_total_mean,
        STDDEV_SAMP(enrol_total) AS enrol_total_std,
        AVG(enrol_minor_ratio) AS enrol_minor_ratio_mean,
        AVG(enrol_adult_ratio) AS enrol_adult_ratio_mean
    FROM enrolment_data_preprocessed
    WHERE district IS NOT NULL
    GROUP BY district
    """

    bio = con.execute(bio_sql).fetchdf()
    enr = con.execute(enr_sql).fetchdf()

    bio["district"] = bio["district"].map(db_to_shp).fillna(bio["district"])
    enr["district"] = enr["district"].map(db_to_shp).fillna(enr["district"])

    bio = bio.groupby("district").mean(numeric_only=True)
    enr = enr.groupby("district").mean(numeric_only=True)

    feat_df = bio.join(enr, how="outer")
    feat_df = feat_df.reindex(districts)

    # Fill missing values column-wise with median, fallback 0
    for col in feat_df.columns:
        med = feat_df[col].median()
        if pd.isna(med):
            med = 0.0
        feat_df[col] = feat_df[col].fillna(med)

    # Standardize columns to zero mean / unit std
    vals = feat_df.values.astype(np.float64)
    mean = vals.mean(axis=0, keepdims=True)
    std = vals.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    vals = (vals - mean) / std

    feat_df = pd.DataFrame(vals, index=feat_df.index, columns=feat_df.columns)
    return feat_df


def build_similarity_matrix(feat_df, epsilon=0.1):
    """
    Cosine similarity matrix on district profile vectors.
    """
    X = feat_df.values.astype(np.float64)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    Xn = X / norms

    W = np.clip(Xn @ Xn.T, 0.0, 1.0)
    W[W < epsilon] = 0.0
    np.fill_diagonal(W, 0.0)

    return W.astype(np.float32)


# ---------------------------------------------------------------------
# COMBINE + LAPLACIAN
# ---------------------------------------------------------------------

def combine_matrices(W_distance, W_similarity, alpha=0.5):
    W = alpha * W_distance + (1.0 - alpha) * W_similarity
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def normalised_laplacian(W):
    """
    L = I - D^{-1/2} W D^{-1/2}
    """
    degree = W.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(len(W), dtype=np.float32) - D_inv_sqrt @ W @ D_inv_sqrt
    return L.astype(np.float32)


# ---------------------------------------------------------------------
# SAVE UTILS
# ---------------------------------------------------------------------

def save_matrix_csv(W, districts, name):
    df = pd.DataFrame(W, index=districts, columns=districts)
    df.to_csv(OUTPUT_DIR / f"{name}.csv")


def save_heatmap(W, districts, name, title):
    n = len(districts)
    figsize = max(12, n * 0.22)

    if n <= 40:
        labels = districts
    else:
        labels = [d[:6] for d in districts]

    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))
    sns.heatmap(
        W,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        cmap="YlOrRd",
        linewidths=0 if n > 60 else 0.1,
        cbar_kws={"shrink": 0.6, "label": "weight"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=90, labelsize=5 if n > 60 else 7)
    ax.tick_params(axis="y", rotation=0, labelsize=5 if n > 60 else 7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(districts, unmatched, feat_df, Wd, Ws, Wc, alpha, sigma2, epsilon):
    lines = [
        "=" * 60,
        "DISTRICT GRAPH BUILD SUMMARY",
        "=" * 60,
        f"Matched districts (N): {len(districts)}",
        f"Unmatched DB districts: {len(unmatched)}",
        f"alpha               : {alpha}",
        f"sigma2              : {sigma2}",
        f"epsilon             : {epsilon}",
        "",
        "Similarity profile features:",
    ]
    lines += [f"  - {c}" for c in feat_df.columns.tolist()]
    lines += [
        "",
        "W_distance:",
        f"  Non-zero entries : {(Wd > 0).sum()}",
        f"  Sparsity         : {100 * (Wd == 0).mean():.2f}%",
        f"  Max weight       : {Wd.max():.4f}",
        "",
        "W_similarity:",
        f"  Non-zero entries : {(Ws > 0).sum()}",
        f"  Sparsity         : {100 * (Ws == 0).mean():.2f}%",
        f"  Max weight       : {Ws.max():.4f}",
        "",
        "W_combined:",
        f"  Non-zero entries : {(Wc > 0).sum()}",
        f"  Sparsity         : {100 * (Wc == 0).mean():.2f}%",
        f"  Max weight       : {Wc.max():.4f}",
        "",
        "Outputs:",
        "  graph_output/W_distance.csv",
        "  graph_output/W_similarity.csv",
        "  graph_output/W_combined.csv",
        "  graph_output/L_normalised_laplacian.csv",
        "  graph_output/district_order.csv",
        "  graph_output/district_mapping.csv",
    ]

    text = "\n".join(lines)
    (OUTPUT_DIR / "summary.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main(shp_path, db_path, alpha, sigma2, epsilon):
    print("=" * 60)
    print("BUILD DISTRICT GRAPH")
    print("=" * 60)
    print(f"alpha={alpha}  sigma2={sigma2}  epsilon={epsilon}")

    con = duckdb.connect(db_path, read_only=True)

    # 1) load shapefile
    print("\n[1] Loading shapefile...")
    gdf, dist_col = load_shapefile(shp_path)
    shp_districts = gdf[dist_col].dropna().astype(str).str.strip().str.title().unique().tolist()
    print(f"  Shapefile districts: {len(shp_districts)}")

    # 2) get DB districts
    print("\n[2] Reading district names from DB...")
    db_districts = fetch_db_districts(con)
    print(f"  DB districts: {len(db_districts)}")

    # 3) fuzzy match
    print("\n[3] Fuzzy matching district names...")
    db_to_shp, unmatched = fuzzy_match_districts(db_districts, shp_districts, FUZZY_THRESH)
    districts = sorted(set(db_to_shp.values()))
    print(f"  Matched districts: {len(districts)}")
    print(f"  Unmatched districts: {len(unmatched)}")

    if len(districts) < 2:
        raise RuntimeError("Too few matched districts to build graph.")

    mapping_df = pd.DataFrame({
        "db_district": list(db_to_shp.keys()),
        "shapefile_district": list(db_to_shp.values())
    }).sort_values(["shapefile_district", "db_district"])
    mapping_df.to_csv(OUTPUT_DIR / "district_mapping.csv", index=False)

    # 4) distance graph
    print("\n[4] Building W_distance...")
    W_distance = build_distance_matrix(
        gdf=gdf,
        dist_col=dist_col,
        districts=districts,
        sigma2=sigma2,
        epsilon=epsilon
    )
    print(f"  W_distance shape: {W_distance.shape}")

    # 5) similarity profile + graph
    print("\n[5] Building similarity profile...")
    feat_df = build_similarity_profile(con, db_to_shp, districts)
    print(f"  Similarity profile shape: {feat_df.shape}")

    print("\n[6] Building W_similarity...")
    W_similarity = build_similarity_matrix(feat_df, epsilon=epsilon)
    print(f"  W_similarity shape: {W_similarity.shape}")

    con.close()

    # 6) combine
    print("\n[7] Combining graphs...")
    W_combined = combine_matrices(W_distance, W_similarity, alpha=alpha)

    # 7) Laplacian
    print("\n[8] Computing normalized Laplacian...")
    L = normalised_laplacian(W_combined)

    # 8) save outputs
    print("\n[9] Saving outputs...")
    save_matrix_csv(W_distance, districts, "W_distance")
    save_matrix_csv(W_similarity, districts, "W_similarity")
    save_matrix_csv(W_combined, districts, "W_combined")
    save_matrix_csv(L, districts, "L_normalised_laplacian")

    pd.Series(districts, name="district").to_csv(
        OUTPUT_DIR / "district_order.csv",
        index=True
    )

    save_heatmap(W_distance, districts, "W_distance", "Distance-based district graph")
    save_heatmap(W_similarity, districts, "W_similarity", "Similarity-based district graph")
    save_heatmap(W_combined, districts, "W_combined", f"Combined district graph (alpha={alpha})")

    save_summary(
        districts=districts,
        unmatched=unmatched,
        feat_df=feat_df,
        Wd=W_distance,
        Ws=W_similarity,
        Wc=W_combined,
        alpha=alpha,
        sigma2=sigma2,
        epsilon=epsilon,
    )

    print("\nDone.")
    print(f"Saved graph files in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build district graph for STGCN")
    parser.add_argument("--shp", required=True, help="Path to India district shapefile")
    parser.add_argument("--db", required=True, help="Path to aadhar.duckdb")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="W_combined = alpha*W_distance + (1-alpha)*W_similarity")
    parser.add_argument("--sigma2", type=float, default=0.1,
                        help="Gaussian spread for distance kernel")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Threshold below which edge weights are set to 0")
    args = parser.parse_args()

    try:
        main(args.shp, args.db, args.alpha, args.sigma2, args.epsilon)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)