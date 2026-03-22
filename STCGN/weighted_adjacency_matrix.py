"""
weighted_adjacency_matrix.py
=============================
Builds a COMBINED weighted adjacency matrix for Indian districts using:
  1. Distance-based weights  (centroid distance between districts)
  2. Enrolment-similarity weights (cosine similarity of feature vectors)
  3. Final W = alpha * W_distance + (1 - alpha) * W_similarity

UPDATED from old version:
  - Uses *_preprocessed tables (not raw tables)
  - Tensor X built from REAL 70 common dates across all 3 tables
    (NO synthetic Gaussian noise — real time series values)
  - Feature vector uses 7 specific STGCN features (not all numeric cols)
  - --T / NOISE_STD / RANDOM_SEED args removed (no longer needed)
  - Tensor C axis order documented and saved to tensor_feature_columns.txt

Outputs (in adjacency_output/):
    W_distance.csv
    W_similarity.csv
    W_combined.csv
    L_normalised_laplacian.csv   ← feed into stgcn_train.py
    feature_tensor_X.npy         ← shape [T=70, N, C=7], feed into stgcn_train.py
    district_order.csv           ← district index → name mapping
    tensor_feature_columns.txt   ← what C=0..6 mean
    summary.txt
    *_heatmap.png

Usage:
    python weighted_adjacency_matrix.py
           --shp Adjacency_marix/2011_Dist.shp
           --db  database/aadhar.duckdb

Dependencies:
    pip install geopandas duckdb numpy pandas scipy matplotlib seaborn rapidfuzz scikit-learn
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rapidfuzz import fuzz, process
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
# Updated: use preprocessed tables
TABLES       = ["biometric_data_preprocessed",
                "demographic_data_preprocessed",
                "enrolment_data_preprocessed"]

FUZZY_THRESH = 85
ALPHA        = 0.5      # distance weight vs similarity weight
SIGMA2       = 0.1      # Gaussian spread for distance kernel
EPSILON      = 0.1      # sparsity threshold — edges below this → 0

# The 7 features that form the C axis of the tensor
# Matches what stgcn_train.py expects
TENSOR_FEATURE_COLS = [
    ("biometric_data_preprocessed",   "bio_age_5_17",    "SUM"),
    ("biometric_data_preprocessed",   "bio_age_17_",     "SUM"),
    ("biometric_data_preprocessed",   "bio_total",       "SUM"),
    ("demographic_data_preprocessed", "demo_age_5_17",   "SUM"),
    ("demographic_data_preprocessed", "demo_age_17_",    "SUM"),
    ("enrolment_data_preprocessed",   "age_5_17",        "SUM"),   # minor proxy
    ("enrolment_data_preprocessed",   "age_18_greater",  "SUM"),   # adult
]
C_NAMES = [
    "bio_age_5_17", "bio_age_17_", "bio_total",
    "demo_age_5_17", "demo_age_17_",
    "enrol_minor", "enrol_adult",
]

OUTPUT_DIR = Path("adjacency_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
RUN_TS   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = OUTPUT_DIR / f"run_{RUN_TS}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Discover common dates across all 3 preprocessed tables
# ══════════════════════════════════════════════════════════════════════════════

def get_common_dates(con) -> list:
    """Returns sorted list of date strings common to all 3 preprocessed tables."""
    sets = []
    for tbl in TABLES:
        rows = con.execute(
            f"SELECT DISTINCT CAST(date AS DATE)::VARCHAR FROM {tbl}"
        ).fetchall()
        sets.append({r[0] for r in rows})
    common = sorted(set.intersection(*sets))
    log.info("Common dates across all 3 tables: %d  (%s → %s)",
             len(common), common[0], common[-1])
    return common


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0b — Introspect DB schema (kept for debugging, uses preprocessed tables)
# ══════════════════════════════════════════════════════════════════════════════

def introspect_db(con) -> dict:
    schema = {}
    log.info("=" * 60)
    log.info("DATABASE SCHEMA — PREPROCESSED TABLES")
    log.info("=" * 60)

    all_tables = con.execute("SHOW TABLES").fetchdf()
    log.info("All tables in DB: %s", all_tables["name"].tolist())

    for tbl in TABLES:
        try:
            info = con.execute(f"DESCRIBE {tbl}").fetchdf()
        except Exception as e:
            log.warning("Could not DESCRIBE %s: %s", tbl, e)
            schema[tbl] = {"columns": [], "numeric_cols": [], "district_col": None}
            continue

        cols     = info["column_name"].tolist()
        dtypes   = info["column_type"].tolist()
        num_cols = [c for c, d in zip(cols, dtypes)
                    if any(t in d.upper() for t in
                           ["INT","FLOAT","DOUBLE","DECIMAL","BIGINT","HUGEINT"])]
        dist_col = next((c for c in cols
                         if c.lower() in ["district","dist_name","district_name"]), None)

        schema[tbl] = {"columns": cols, "numeric_cols": num_cols, "district_col": dist_col}
        log.info("  [%s]  district_col=%s  numeric_cols=%d",
                 tbl, dist_col, len(num_cols))
        sample = con.execute(f"SELECT * FROM {tbl} LIMIT 2").fetchdf()
        log.info("  Sample:\n%s", sample.to_string())

    return schema


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load shapefile + fuzzy-match
# ══════════════════════════════════════════════════════════════════════════════

def normalise(s):
    return " ".join(str(s).lower().split())

def load_shapefile(shp_path):
    log.info("Loading shapefile: %s", shp_path)
    gdf = gpd.read_file(shp_path)
    dist_col = next(
        (c for c in ["DISTRICT","district","District","DISTNAME","dtname"]
         if c in gdf.columns), None)
    if dist_col is None:
        raise ValueError(f"No district column found. Cols: {list(gdf.columns)}")
    gdf[dist_col] = gdf[dist_col].str.strip().str.title()
    log.info("Shapefile: %d rows  district_col='%s'", len(gdf), dist_col)
    return gdf, dist_col

def fuzzy_match(db_districts, shp_districts):
    shp_norm = {normalise(s): s for s in shp_districts}
    matched, unmatched = {}, []
    for d in db_districts:
        res = process.extractOne(normalise(d), list(shp_norm.keys()),
                                 scorer=fuzz.WRatio)
        if res and res[1] >= FUZZY_THRESH:
            matched[d] = shp_norm[res[0]]
        else:
            unmatched.append(d)
    log.info("  Matched %d / %d  |  Unmatched: %d",
             len(matched), len(db_districts), len(unmatched))
    return matched, unmatched


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build W_distance
# ══════════════════════════════════════════════════════════════════════════════

def build_distance_matrix(gdf, dist_col, districts):
    log.info("Computing centroid distances for %d districts ...", len(districts))
    sub = gdf[gdf[dist_col].isin(districts)].copy()
    sub = sub.dissolve(by=dist_col).reset_index()
    sub = sub.set_index(dist_col).reindex(districts)
    if sub.crs and sub.crs.is_geographic:
        sub = sub.to_crs("EPSG:32643")
    centroids = sub.geometry.centroid
    coords    = np.array([[c.x, c.y] for c in centroids])
    D         = cdist(coords, coords, metric="euclidean")
    d_max     = D.max()
    D_norm    = D / d_max if d_max > 0 else D
    W         = np.exp(-(D_norm ** 2) / SIGMA2)
    W[W < EPSILON] = 0
    np.fill_diagonal(W, 0)
    log.info("  W_distance: sparsity=%.1f%%  max=%.4f",
             100*(W==0).mean(), W.max())
    return W.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build profile feature df for W_similarity
#          Uses monthly dates only (stable signal for similarity)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_df(con, districts, db_to_shp, all_dates):
    """
    Updated: queries preprocessed tables, uses monthly dates for stable profile.
    Returns normalised [N, C] dataframe for cosine similarity.
    """
    monthly = [d for d in all_dates if d.endswith("-01")]
    if not monthly:
        monthly = all_dates[:4]   # fallback: first 4 dates
    log.info("Profile dates (monthly): %s", monthly)
    dts = ", ".join([f"'{d}'" for d in monthly])

    frames = []
    for tbl in TABLES:
        # Get numeric cols from preprocessed table
        info     = con.execute(f"DESCRIBE {tbl}").fetchdf()
        all_cols = info["column_name"].tolist()
        all_dtys = info["column_type"].tolist()
        num_cols = [c for c, d in zip(all_cols, all_dtys)
                    if any(t in d.upper() for t in
                           ["INT","FLOAT","DOUBLE","DECIMAL","BIGINT","HUGEINT"])
                    and c not in ["id","pincode","year","month","quarter",
                                  "day_of_week","is_weekend","day_of_year"]]

        if not num_cols:
            continue

        agg  = ", ".join([f'SUM("{c}") AS "{tbl}__{c}"' for c in num_cols])
        sql  = f"""
            SELECT district, {agg}
            FROM {tbl}
            WHERE CAST(date AS DATE)::VARCHAR IN ({dts})
              AND district IS NOT NULL
            GROUP BY district
        """
        df = con.execute(sql).fetchdf()
        df["district"] = df["district"].map(db_to_shp).fillna(df["district"])
        df = df.groupby("district").sum()
        frames.append(df)
        log.info("  [%s] %d rows × %d feature cols", tbl, len(df), len(num_cols))

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.join(df, how="outer")
    merged = merged.reindex(districts).fillna(0)

    scaler = MinMaxScaler()
    merged[:] = scaler.fit_transform(merged.values)
    log.info("Profile feature matrix: %s", merged.shape)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build W_similarity
# ══════════════════════════════════════════════════════════════════════════════

def build_similarity_matrix(feat_df):
    X     = feat_df.values.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    X_n   = X / norms
    W     = np.clip(X_n @ X_n.T, 0, 1)
    W[W < EPSILON] = 0
    np.fill_diagonal(W, 0)
    log.info("  W_similarity: sparsity=%.1f%%  max=%.4f",
             100*(W==0).mean(), W.max())
    return W.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Combine + Laplacian
# ══════════════════════════════════════════════════════════════════════════════

def combine_matrices(W_dist, W_sim, alpha=ALPHA):
    W = alpha * W_dist + (1 - alpha) * W_sim
    np.fill_diagonal(W, 0)
    log.info("  W_combined (alpha=%.2f): sparsity=%.1f%%  max=%.4f",
             alpha, 100*(W==0).mean(), W.max())
    return W.astype(np.float32)

def normalised_laplacian(W):
    """L = I - D^{-1/2} W D^{-1/2}  (STGCN Eq. 2)"""
    degree     = W.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(len(W), dtype=np.float32) - D_inv_sqrt @ W @ D_inv_sqrt
    return L.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Build REAL feature tensor X [T=70, N, C=7]
#          UPDATED: real values from preprocessed tables — NO synthetic noise
# ══════════════════════════════════════════════════════════════════════════════

def build_tensor(con, districts, db_to_shp, all_dates):
    """
    For each of the 70 common dates, fetches the 7 STGCN features per district
    directly from the preprocessed tables.

    C axis order (matches stgcn_train.py):
      0: bio_age_5_17
      1: bio_age_17_
      2: bio_total
      3: demo_age_5_17
      4: demo_age_17_
      5: enrol_minor  (age_5_17 from enrolment table)
      6: enrol_adult  (age_18_greater)
    """
    log.info("Building real tensor X [T=%d, N=%d, C=7] — NO synthetic noise",
             len(all_dates), len(districts))
    dts = ", ".join([f"'{d}'" for d in all_dates])

    # Fetch bio features
    bio = con.execute(f"""
        SELECT CAST(date AS DATE)::VARCHAR AS date, district,
            SUM(bio_age_5_17) AS bio_age_5_17,
            SUM(bio_age_17_)  AS bio_age_17_,
            SUM(bio_total)    AS bio_total
        FROM biometric_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts})
          AND district IS NOT NULL
        GROUP BY date, district
    """).fetchdf()

    # Fetch demo features
    demo = con.execute(f"""
        SELECT CAST(date AS DATE)::VARCHAR AS date, district,
            SUM(demo_age_5_17) AS demo_age_5_17,
            SUM(demo_age_17_)  AS demo_age_17_
        FROM demographic_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts})
          AND district IS NOT NULL
        GROUP BY date, district
    """).fetchdf()

    # Fetch enrolment features
    enrol = con.execute(f"""
        SELECT CAST(date AS DATE)::VARCHAR AS date, district,
            SUM(age_5_17)         AS enrol_minor,
            SUM(age_18_greater)   AS enrol_adult
        FROM enrolment_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts})
          AND district IS NOT NULL
        GROUP BY date, district
    """).fetchdf()

    # Map district names → shapefile names
    bio["district"] = bio["district"].map(db_to_shp).fillna(bio["district"])
    demo["district"] = demo["district"].map(db_to_shp).fillna(demo["district"])
    enrol["district"] = enrol["district"].map(db_to_shp).fillna(enrol["district"])

    # Aggregate duplicates
    bio = bio.groupby(["date", "district"]).sum().reset_index()
    demo = demo.groupby(["date", "district"]).sum().reset_index()
    enrol = enrol.groupby(["date", "district"]).sum().reset_index()

    # Merge all on date + district
    merged = bio.merge(demo,  on=["date","district"], how="outer")
    merged = merged.merge(enrol, on=["date","district"], how="outer")
    merged = merged.fillna(0)

    feat_cols = ["bio_age_5_17","bio_age_17_","bio_total",
                 "demo_age_5_17","demo_age_17_",
                 "enrol_minor","enrol_adult"]

    # Global min-max normalise each feature to [0,1]
    scaler = MinMaxScaler()
    merged[feat_cols] = scaler.fit_transform(merged[feat_cols])

    # Build [T, N, C] tensor
    T = len(all_dates)
    N = len(districts)
    C = len(feat_cols)
    X = np.zeros((T, N, C), dtype=np.float32)

    dist_idx = {d: i for i, d in enumerate(districts)}
    for t, date in enumerate(all_dates):
        slice_df = merged[merged["date"] == date]
        for _, row in slice_df.iterrows():
            ni = dist_idx.get(row["district"])
            if ni is not None:
                X[t, ni, :] = row[feat_cols].values.astype(np.float32)

    # Coverage report
    filled = (X.sum(axis=2) > 0).sum()
    total  = T * N
    log.info("  Tensor filled: %d / %d slots (%.1f%%)",
             filled, total, 100*filled/total)
    log.info("  Tensor X shape: %s  dtype=%s", X.shape, X.dtype)
    return X, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_matrix_csv(W, districts, name):
    pd.DataFrame(W, index=districts, columns=districts).to_csv(
        OUTPUT_DIR / f"{name}.csv")
    log.info("Saved → %s.csv", name)

def save_heatmap(W, districts, name, title):
    n     = len(districts)
    fsz   = max(12, n * 0.25)
    lbls  = districts if n <= 40 else [d[:6] for d in districts]
    fig, ax = plt.subplots(figsize=(fsz, fsz * 0.85))
    sns.heatmap(W, ax=ax,
                xticklabels=lbls, yticklabels=lbls,
                cmap="YlOrRd", vmin=0, vmax=W.max(),
                linewidths=0 if n > 60 else 0.1,
                cbar_kws={"shrink":0.6,"label":"weight"})
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=90, labelsize=5 if n>60 else 7)
    ax.tick_params(axis="y", rotation=0,  labelsize=5 if n>60 else 7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap → %s_heatmap.png", name)

def save_summary(districts, W_dist, W_sim, W_comb, X, all_dates, feat_cols):
    lines = [
        "=" * 60,
        "ADJACENCY MATRIX BUILD SUMMARY",
        "=" * 60,
        f"Districts (N)         : {len(districts)}",
        f"Time steps (T)        : {X.shape[0]}  ← REAL dates (no synthetic noise)",
        f"Feature channels (C)  : {X.shape[2]}",
        f"Tensor shape [T,N,C]  : {X.shape}",
        f"Date range            : {all_dates[0]} → {all_dates[-1]}",
        "",
        "W_distance:",
        f"  Non-zero edges : {(W_dist>0).sum()//2}",
        f"  Sparsity       : {100*(W_dist==0).mean():.1f}%",
        f"  Max weight     : {W_dist.max():.4f}",
        "",
        "W_similarity:",
        f"  Non-zero edges : {(W_sim>0).sum()//2}",
        f"  Sparsity       : {100*(W_sim==0).mean():.1f}%",
        f"  Max weight     : {W_sim.max():.4f}",
        "",
        "W_combined:",
        f"  Non-zero edges : {(W_comb>0).sum()//2}",
        f"  Sparsity       : {100*(W_comb==0).mean():.1f}%",
        f"  Max weight     : {W_comb.max():.4f}",
        "",
        "Tensor C-axis features:",
    ] + [f"  C={i}: {c}" for i, c in enumerate(feat_cols)] + [
        "",
        "Feed into stgcn_train.py:",
        "  --tensor     adjacency_output/feature_tensor_X.npy",
        "  --laplacian  adjacency_output/L_normalised_laplacian.csv",
        "  --districts  adjacency_output/district_order.csv",
    ]
    txt = "\n".join(lines)
    log.info("\n" + txt)
    (OUTPUT_DIR / "summary.txt").write_text(txt,encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(shp_path, db_path, alpha=ALPHA, sigma2=SIGMA2, epsilon=EPSILON):
    global ALPHA, SIGMA2, EPSILON
    ALPHA, SIGMA2, EPSILON = alpha, sigma2, epsilon

    log.info("=" * 60)
    log.info("WEIGHTED ADJACENCY MATRIX BUILDER")
    log.info("Using PREPROCESSED tables + REAL time series (no synthetic noise)")
    log.info("alpha=%.2f  sigma2=%.3f  epsilon=%.3f", ALPHA, SIGMA2, EPSILON)
    log.info("=" * 60)

    con = duckdb.connect(db_path, read_only=True)

    # Step 0: common dates + schema
    log.info("\n── Step 0: Dates + schema ──")
    all_dates = get_common_dates(con)
    introspect_db(con)

    # Step 1: shapefile + district matching
    log.info("\n── Step 1: Shapefile + fuzzy match ──")
    gdf, dist_col = load_shapefile(shp_path)
    shp_districts = gdf[dist_col].dropna().unique().tolist()

    all_db = set()
    for tbl in TABLES:
        rows = con.execute(
            f"SELECT DISTINCT district FROM {tbl} WHERE district IS NOT NULL"
        ).fetchall()
        all_db.update(r[0] for r in rows)
    log.info("Unique DB districts (preprocessed): %d", len(all_db))

    db_to_shp, unmatched = fuzzy_match(list(all_db), shp_districts)
    districts = sorted(set(db_to_shp.values()))
    log.info("Final matched districts: %d", len(districts))
    if len(districts) < 2:
        log.error("Too few districts matched.")
        sys.exit(1)

    # Step 2: distance matrix
    log.info("\n── Step 2: W_distance ──")
    W_dist = build_distance_matrix(gdf, dist_col, districts)

    # Step 3: profile df for similarity
    log.info("\n── Step 3: Profile feature df ──")
    feat_df = build_feature_df(con, districts, db_to_shp, all_dates)

    # Step 4: similarity matrix
    log.info("\n── Step 4: W_similarity ──")
    W_sim = build_similarity_matrix(feat_df)

    # Step 5: combined + Laplacian
    log.info("\n── Step 5: W_combined + Laplacian ──")
    W_comb = combine_matrices(W_dist, W_sim, ALPHA)
    L      = normalised_laplacian(W_comb)

    # Step 6: REAL tensor (no synthetic noise)
    log.info("\n── Step 6: Feature tensor X (REAL data, T=%d) ──", len(all_dates))
    X, feat_cols = build_tensor(con, districts, db_to_shp, all_dates)

    con.close()

    # Step 7: save
    log.info("\n── Step 7: Saving outputs ──")
    save_matrix_csv(W_dist,  districts, "W_distance")
    save_matrix_csv(W_sim,   districts, "W_similarity")
    save_matrix_csv(W_comb,  districts, "W_combined")
    save_matrix_csv(L,       districts, "L_normalised_laplacian")

    save_heatmap(W_dist,  districts, "W_distance",   "Distance-based weights")
    save_heatmap(W_sim,   districts, "W_similarity", "Enrolment-similarity weights")
    save_heatmap(W_comb,  districts, "W_combined",   f"Combined weights (α={ALPHA})")

    np.save(OUTPUT_DIR / "feature_tensor_X.npy", X)
    log.info("Saved tensor X %s → feature_tensor_X.npy", X.shape)

    pd.Series(districts, name="district").to_csv(
        OUTPUT_DIR / "district_order.csv", index=True)

    (OUTPUT_DIR / "tensor_feature_columns.txt").write_text(
        "\n".join([f"C={i}: {c}" for i, c in enumerate(feat_cols)]))

    save_summary(districts, W_dist, W_sim, W_comb, X, all_dates, feat_cols)

    log.info("\n" + "="*60)
    log.info("DONE — adjacency_output/")
    log.info("Tensor: %s  (REAL data, %d dates)", X.shape, len(all_dates))
    log.info("Next: python stgcn_train.py")
    log.info("="*60)

    print(f"""
╔══════════════════════════════════════════════════════╗
║         HOW TO USE THESE OUTPUTS IN STGCN            ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Tensor X shape : {str(X.shape):<34}║
║  T = {len(all_dates)} REAL dates (no synthetic noise)          ║
║  N = {len(districts)} matched districts                        ║
║  C = {len(feat_cols)} features (see tensor_feature_columns.txt)║
║                                                      ║
║  Run STGCN:                                          ║
║    python stgcn_train.py                             ║
║      --tensor    adjacency_output/feature_tensor_X.npy║
║      --laplacian adjacency_output/L_normalised_laplacian.csv║
║      --districts adjacency_output/district_order.csv ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build combined adjacency matrix + real tensor for STGCN."
    )
    p.add_argument("--shp",     required=True,
                   help="Path to 2011_Dist.shp (e.g. Adjacency_marix/2011_Dist.shp)")
    p.add_argument("--db",      required=True,
                   help="Path to aadhar.duckdb")
    p.add_argument("--alpha",   type=float, default=ALPHA,
                   help=f"alpha*W_dist + (1-alpha)*W_sim  (default: {ALPHA})")
    p.add_argument("--sigma2",  type=float, default=SIGMA2,
                   help=f"Gaussian spread for distance kernel  (default: {SIGMA2})")
    p.add_argument("--epsilon", type=float, default=EPSILON,
                   help=f"Sparsity threshold  (default: {EPSILON})")
    a = p.parse_args()
    main(a.shp, a.db, a.alpha, a.sigma2, a.epsilon)