"""
weighted_adjacency_matrix.py
=============================
Builds a COMBINED weighted adjacency matrix for Indian districts using:
  1. Distance-based weights  (centroid distance between districts, like STGCN PeMSD7)
  2. Enrolment-similarity weights (cosine similarity of feature vectors)
  3. Final W = alpha * W_distance + (1 - alpha) * W_similarity

Also:
  - Introspects all 3 DuckDB tables to discover columns automatically
  - Prints full schema so you know what columns exist
  - Engineers synthetic time dimension from the single snapshot (1/3/2025)
    by adding Gaussian noise -> creates T=12 pseudo-monthly steps per district
  - Saves W_distance, W_similarity, W_combined as CSVs + heatmap PNGs
  - Saves the final feature tensor X of shape [T x N x C] as .npy

Usage
-----
python weighted_adjacency_matrix.py --shp path/to/2011_Dist.shp --db path/to/your.duckdb

Dependencies
------------
pip install geopandas duckdb numpy pandas scipy matplotlib seaborn rapidfuzz scikit-learn
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from rapidfuzz import fuzz, process
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

# ── Config ─────────────────────────────────────────────────────────────────────
TABLES          = ["biometric_data", "demographic_data", "enrolment_data"]
FUZZY_THRESH    = 85          # RapidFuzz match threshold for district name alignment
ALPHA           = 0.5         # Weight: 0.5 * distance + 0.5 * similarity
SIGMA2          = 0.1         # Spread for distance Gaussian (after normalisation)
EPSILON         = 0.1         # Sparsity threshold — edges below this become 0
T_STEPS         = 12          # Synthetic time steps generated from single snapshot
NOISE_STD       = 0.03        # Std of Gaussian noise for synthetic time steps
RANDOM_SEED     = 42

OUTPUT_DIR = Path("adjacency_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
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
# STEP 0 — Introspect DB: discover columns in every table
# ══════════════════════════════════════════════════════════════════════════════

def introspect_db(con: duckdb.DuckDBPyConnection) -> dict:
    """
    Print schema of every table and return a dict:
      { table_name: { "columns": [...], "numeric_cols": [...], "district_col": str } }
    """
    schema = {}
    log.info("=" * 60)
    log.info("DATABASE SCHEMA INSPECTION")
    log.info("=" * 60)

    # List all tables
    all_tables = con.execute("SHOW TABLES").fetchdf()
    log.info("Tables in DB: %s", all_tables["name"].tolist())

    for tbl in TABLES:
        try:
            info = con.execute(f"DESCRIBE {tbl}").fetchdf()
        except Exception as e:
            log.warning("  Could not DESCRIBE %s: %s", tbl, e)
            schema[tbl] = {"columns": [], "numeric_cols": [], "district_col": None}
            continue

        cols      = info["column_name"].tolist()
        dtypes    = info["column_type"].tolist()
        num_cols  = [c for c, d in zip(cols, dtypes)
                     if any(t in d.upper() for t in
                            ["INT", "FLOAT", "DOUBLE", "DECIMAL", "BIGINT", "HUGEINT"])]

        # Detect district column (common names)
        dist_col = None
        for candidate in ["district", "District", "DISTRICT", "dist_name", "district_name"]:
            if candidate in cols:
                dist_col = candidate
                break

        schema[tbl] = {
            "columns":      cols,
            "numeric_cols": num_cols,
            "district_col": dist_col,
        }

        log.info("")
        log.info("  Table: %s", tbl)
        log.info("  Columns: %s", cols)
        log.info("  Numeric cols: %s", num_cols)
        log.info("  District col: %s", dist_col)

        # Show first 3 rows
        sample = con.execute(f"SELECT * FROM {tbl} LIMIT 3").fetchdf()
        log.info("  Sample rows:\n%s", sample.to_string())

    return schema


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load shapefile + fuzzy-match districts
# ══════════════════════════════════════════════════════════════════════════════

def normalise(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return " ".join(name.lower().split())


def fuzzy_match(db_districts: list, shp_districts: list) -> tuple:
    """Returns (matched_dict {db->shp}, unmatched_list)"""
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


def load_shapefile(shp_path: str) -> tuple:
    """Returns (dissolved GeoDataFrame, district_col_name)"""
    log.info("Loading shapefile: %s", shp_path)
    gdf = gpd.read_file(shp_path)

    dist_col = None
    for c in ["DISTRICT", "district", "District", "DISTNAME", "dtname"]:
        if c in gdf.columns:
            dist_col = c
            break
    if dist_col is None:
        raise ValueError(f"No district column found. Cols: {list(gdf.columns)}")

    gdf[dist_col] = gdf[dist_col].str.strip().str.title()
    log.info("Shapefile: %d rows  |  district col: '%s'", len(gdf), dist_col)
    return gdf, dist_col


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build W_distance  (Gaussian on normalised centroid distances)
# ══════════════════════════════════════════════════════════════════════════════

def build_distance_matrix(gdf: gpd.GeoDataFrame,
                           dist_col: str,
                           districts: list) -> np.ndarray:
    """
    For each pair of districts compute centroid distance,
    then apply Gaussian kernel: w = exp(-d^2 / sigma2), zero if < epsilon.
    Distances are normalised to [0,1] before applying the kernel so that
    sigma2 and epsilon are dataset-agnostic.
    """
    log.info("  Computing centroid distances for %d districts ...", len(districts))

    # Dissolve so multi-polygon districts collapse to one centroid
    sub = gdf[gdf[dist_col].isin(districts)].copy()
    sub = sub.dissolve(by=dist_col).reset_index()
    sub = sub.set_index(dist_col).reindex(districts)  # keep user-defined order

    # Project to a metric CRS for accurate km distances
    if sub.crs and sub.crs.is_geographic:
        sub = sub.to_crs("EPSG:32643")   # UTM zone 43N — covers India

    centroids = sub.geometry.centroid
    coords = np.array([[c.x, c.y] for c in centroids])

    # Pairwise Euclidean distance
    D = cdist(coords, coords, metric="euclidean")

    # Normalise distances to [0, 1]
    d_max = D.max()
    if d_max > 0:
        D_norm = D / d_max
    else:
        D_norm = D

    # Gaussian kernel
    W = np.exp(-(D_norm ** 2) / SIGMA2)
    W[W < EPSILON] = 0
    np.fill_diagonal(W, 0)

    log.info("  W_distance: sparsity=%.1f%%  min=%.4f  max=%.4f",
             100 * (W == 0).mean(), W[W > 0].min() if (W > 0).any() else 0, W.max())
    return W.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build feature dataframe per district (across all 3 tables)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_df(con: duckdb.DuckDBPyConnection,
                     schema: dict,
                     districts: list,
                     db_to_shp: dict) -> pd.DataFrame:
    """
    For each table, aggregate numeric columns per district (sum / mean).
    Joins all tables into one wide dataframe indexed by district.
    Returns normalised feature matrix of shape [N, total_numeric_features].
    """
    # Reverse map: shapefile name -> DB name (for filtering)
    shp_to_db = {v: k for k, v in db_to_shp.items()}
    db_district_names = [shp_to_db.get(d, d) for d in districts]

    frames = []
    for tbl, info in schema.items():
        dist_col  = info["district_col"]
        num_cols  = info["numeric_cols"]

        if dist_col is None or not num_cols:
            log.warning("  [%s] No district col or no numeric cols — skipping", tbl)
            continue

        # Build query: aggregate numeric cols per district
        agg_exprs = ", ".join([f'SUM("{c}") AS "{tbl}_{c}"' for c in num_cols])
        placeholders = ", ".join(["?" for _ in db_district_names])
        query = (f'SELECT "{dist_col}", {agg_exprs} '
                 f'FROM {tbl} '
                 f'WHERE "{dist_col}" IN ({placeholders}) '
                 f'GROUP BY "{dist_col}"')

        try:
            df = con.execute(query, db_district_names).fetchdf()
            df = df.rename(columns={dist_col: "district"})
            frames.append(df)
            log.info("  [%s] fetched %d rows, %d feature cols",
                     tbl, len(df), len(num_cols))
        except Exception as e:
            log.error("  [%s] query failed: %s", tbl, e)

    if not frames:
        raise RuntimeError("No feature data could be extracted from DB.")

    # Merge all tables on district
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="district", how="outer")

    # Map DB district names back to shapefile names for alignment
    merged["district"] = merged["district"].map(db_to_shp).fillna(merged["district"])
    merged = merged.set_index("district").reindex(districts).fillna(0)

    # Normalise every feature to [0, 1]
    scaler = MinMaxScaler()
    feat_cols = merged.columns.tolist()
    merged[feat_cols] = scaler.fit_transform(merged[feat_cols])

    log.info("Feature matrix shape: %s  |  cols: %s", merged.shape, feat_cols[:8])
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build W_similarity  (cosine similarity of feature vectors)
# ══════════════════════════════════════════════════════════════════════════════

def build_similarity_matrix(feat_df: pd.DataFrame) -> np.ndarray:
    """
    Cosine similarity between district feature vectors.
    Negative values clipped to 0.  Diagonal set to 0.
    """
    X = feat_df.values.astype(np.float64)

    # Handle zero vectors (all-zero districts) to avoid NaN
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    X_norm = X / norms

    W = X_norm @ X_norm.T          # cosine similarity in [−1, 1]
    W = np.clip(W, 0, 1)           # keep only positive similarity
    W[W < EPSILON] = 0
    np.fill_diagonal(W, 0)

    log.info("  W_similarity: sparsity=%.1f%%  min=%.4f  max=%.4f",
             100 * (W == 0).mean(), W[W > 0].min() if (W > 0).any() else 0, W.max())
    return W.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Combine: W = alpha * W_dist + (1-alpha) * W_sim
# ══════════════════════════════════════════════════════════════════════════════

def combine_matrices(W_dist: np.ndarray,
                     W_sim: np.ndarray,
                     alpha: float = ALPHA) -> np.ndarray:
    W = alpha * W_dist + (1 - alpha) * W_sim
    np.fill_diagonal(W, 0)
    log.info("  W_combined (alpha=%.2f): sparsity=%.1f%%  max=%.4f",
             alpha, 100 * (W == 0).mean(), W.max())
    return W.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Feature Engineering: single snapshot → T synthetic time steps
# ══════════════════════════════════════════════════════════════════════════════

def build_tensor(feat_df: pd.DataFrame,
                 t_steps: int = T_STEPS,
                 noise_std: float = NOISE_STD) -> np.ndarray:
    """
    Since data is a single snapshot (1/3/2025), we simulate T monthly steps
    by adding small Gaussian noise to the base feature vector.
    This is standard practice for single-snapshot graph learning tasks.

    Output tensor X shape: [T, N, C]
      T = t_steps  (synthetic months)
      N = number of districts
      C = number of feature columns
    """
    rng = np.random.default_rng(RANDOM_SEED)
    base = feat_df.values.astype(np.float32)   # [N, C]
    N, C = base.shape

    X = np.zeros((t_steps, N, C), dtype=np.float32)
    for t in range(t_steps):
        noise = rng.normal(0, noise_std, size=(N, C)).astype(np.float32)
        X[t] = np.clip(base + noise, 0, 1)

    log.info("Feature tensor X shape: %s  [T=%d, N=%d, C=%d]",
             X.shape, t_steps, N, C)
    return X


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Normalised Laplacian (needed for STGCN graph conv)
# ══════════════════════════════════════════════════════════════════════════════

def normalised_laplacian(W: np.ndarray) -> np.ndarray:
    """
    L = I - D^{-1/2} W D^{-1/2}
    This is exactly what the STGCN paper uses (Eq. 2).
    """
    degree = W.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt
    return L.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_matrix_csv(W: np.ndarray, districts: list, name: str):
    path = OUTPUT_DIR / f"{name}.csv"
    pd.DataFrame(W, index=districts, columns=districts).to_csv(path)
    log.info("Saved %s → %s", name, path)


def save_heatmap(W: np.ndarray, districts: list, name: str, title: str):
    """Save a heatmap PNG. Truncates labels if > 40 districts for readability."""
    path = OUTPUT_DIR / f"{name}_heatmap.png"
    n = len(districts)
    fig_size = max(12, n * 0.25)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    labels = districts if n <= 40 else [d[:6] for d in districts]
    sns.heatmap(
        W, ax=ax,
        xticklabels=labels, yticklabels=labels,
        cmap="YlOrRd", vmin=0, vmax=W.max(),
        linewidths=0 if n > 60 else 0.1,
        cbar_kws={"shrink": 0.6, "label": "weight"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=90, labelsize=5 if n > 60 else 7)
    ax.tick_params(axis="y", rotation=0,  labelsize=5 if n > 60 else 7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap → %s", path)


def save_tensor(X: np.ndarray, feat_cols: list):
    path = OUTPUT_DIR / "feature_tensor_X.npy"
    np.save(path, X)
    log.info("Saved tensor X %s → %s", X.shape, path)

    # Also save column names so you know what C=0, C=1, ... mean
    cols_path = OUTPUT_DIR / "tensor_feature_columns.txt"
    cols_path.write_text("\n".join(feat_cols))
    log.info("Tensor column names → %s", cols_path)


def save_summary(districts: list, W_dist, W_sim, W_comb, X, feat_df):
    """Print and save a human-readable summary of everything built."""
    lines = [
        "=" * 60,
        "ADJACENCY MATRIX BUILD SUMMARY",
        "=" * 60,
        f"Districts (N)         : {len(districts)}",
        f"Feature columns (C)   : {len(feat_df.columns)}",
        f"Time steps (T)        : {X.shape[0]}",
        f"Tensor shape [T,N,C]  : {X.shape}",
        "",
        "W_distance:",
        f"  Non-zero edges : {(W_dist > 0).sum() // 2}",
        f"  Sparsity       : {100*(W_dist==0).mean():.1f}%",
        f"  Max weight     : {W_dist.max():.4f}",
        "",
        "W_similarity:",
        f"  Non-zero edges : {(W_sim > 0).sum() // 2}",
        f"  Sparsity       : {100*(W_sim==0).mean():.1f}%",
        f"  Max weight     : {W_sim.max():.4f}",
        "",
        "W_combined:",
        f"  Non-zero edges : {(W_comb > 0).sum() // 2}",
        f"  Sparsity       : {100*(W_comb==0).mean():.1f}%",
        f"  Max weight     : {W_comb.max():.4f}",
        "",
        "Feature columns used:",
    ] + [f"  {i:>3}. {c}" for i, c in enumerate(feat_df.columns)]

    summary = "\n".join(lines)
    log.info("\n" + summary)
    (OUTPUT_DIR / "summary.txt").write_text(summary)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(shp_path: str, db_path: str):
    log.info("=" * 60)
    log.info("WEIGHTED ADJACENCY MATRIX BUILDER")
    log.info("alpha=%.2f  sigma2=%.3f  epsilon=%.3f  T=%d",
             ALPHA, SIGMA2, EPSILON, T_STEPS)
    log.info("=" * 60)

    # ── Connect to DuckDB ──────────────────────────────────────────────────
    log.info("Connecting to DuckDB: %s", db_path)
    con = duckdb.connect(db_path, read_only=True)

    # ── STEP 0: Inspect DB ─────────────────────────────────────────────────
    schema = introspect_db(con)

    # ── STEP 1: Load shapefile ─────────────────────────────────────────────
    gdf, dist_col = load_shapefile(shp_path)
    shp_districts = gdf[dist_col].dropna().unique().tolist()

    # Collect all districts appearing in any table
    all_db_districts = set()
    for tbl, info in schema.items():
        dc = info["district_col"]
        if dc:
            rows = con.execute(
                f'SELECT DISTINCT "{dc}" FROM {tbl} WHERE "{dc}" IS NOT NULL'
            ).fetchall()
            all_db_districts.update(r[0] for r in rows)

    log.info("Unique districts across all tables: %d", len(all_db_districts))

    # Fuzzy match DB → shapefile
    log.info("Fuzzy-matching districts to shapefile ...")
    db_to_shp, unmatched = fuzzy_match(list(all_db_districts), shp_districts)

    # Final ordered district list (shapefile names)
    districts = sorted(set(db_to_shp.values()))
    log.info("Final district count for matrices: %d", len(districts))

    if len(districts) < 2:
        log.error("Too few districts matched — check shapefile and DB data.")
        sys.exit(1)

    # ── STEP 2: Distance matrix ────────────────────────────────────────────
    log.info("")
    log.info("─ Building W_distance ─────────────────────────────────────")
    W_dist = build_distance_matrix(gdf, dist_col, districts)

    # ── STEP 3: Feature dataframe ──────────────────────────────────────────
    log.info("")
    log.info("─ Building feature dataframe ──────────────────────────────")
    feat_df = build_feature_df(con, schema, districts, db_to_shp)

    # ── STEP 4: Similarity matrix ──────────────────────────────────────────
    log.info("")
    log.info("─ Building W_similarity ───────────────────────────────────")
    W_sim = build_similarity_matrix(feat_df)

    # ── STEP 5: Combined matrix ────────────────────────────────────────────
    log.info("")
    log.info("─ Combining matrices (alpha=%.2f) ─────────────────────────", ALPHA)
    W_comb = combine_matrices(W_dist, W_sim, ALPHA)

    # ── STEP 6: Normalised Laplacian ───────────────────────────────────────
    log.info("")
    log.info("─ Computing normalised Laplacian ──────────────────────────")
    L = normalised_laplacian(W_comb)

    # ── STEP 7: Feature tensor ─────────────────────────────────────────────
    log.info("")
    log.info("─ Building feature tensor X [T x N x C] ──────────────────")
    X = build_tensor(feat_df, T_STEPS, NOISE_STD)

    # ── STEP 8: Save everything ────────────────────────────────────────────
    log.info("")
    log.info("─ Saving outputs ──────────────────────────────────────────")
    save_matrix_csv(W_dist,  districts, "W_distance")
    save_matrix_csv(W_sim,   districts, "W_similarity")
    save_matrix_csv(W_comb,  districts, "W_combined")
    save_matrix_csv(L,       districts, "L_normalised_laplacian")

    save_heatmap(W_dist,  districts, "W_distance",    "Distance-based weights")
    save_heatmap(W_sim,   districts, "W_similarity",  "Enrolment-similarity weights")
    save_heatmap(W_comb,  districts, "W_combined",    f"Combined weights (α={ALPHA})")

    save_tensor(X, feat_df.columns.tolist())
    save_summary(districts, W_dist, W_sim, W_comb, X, feat_df)

    con.close()

    log.info("")
    log.info("=" * 60)
    log.info("DONE. All outputs in: %s", OUTPUT_DIR.resolve())
    log.info("=" * 60)

    # Print quick usage guide for STGCN
    print("""
╔══════════════════════════════════════════════════════╗
║         HOW TO USE THESE OUTPUTS IN STGCN            ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  import numpy as np                                  ║
║  import pandas as pd                                 ║
║                                                      ║
║  # Adjacency matrix (weighted)                       ║
║  W = pd.read_csv("adjacency_output/W_combined.csv",  ║
║          index_col=0).values                         ║
║                                                      ║
║  # Normalised Laplacian for graph conv               ║
║  L = pd.read_csv("adjacency_output/              ║
║          L_normalised_laplacian.csv",                ║
║          index_col=0).values                         ║
║                                                      ║
║  # Feature tensor [T, N, C]                          ║
║  X = np.load("adjacency_output/feature_tensor_X.npy")║
║  # X.shape → (12, N_districts, C_features)           ║
║                                                      ║
║  # Feed W and X into your STGCN model                ║
╚══════════════════════════════════════════════════════╝
""")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build combined weighted adjacency matrix for STGCN on Aadhaar data."
    )
    parser.add_argument(
        "--shp", required=True,
        help="Path to 2011_Dist.shp (e.g. Adjacency_marix/2011_Dist.shp)"
    )
    parser.add_argument(
        "--db", required=True,
        help="Path to your DuckDB file (e.g. database/aadhaar.duckdb)"
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA,
        help=f"Mix ratio: alpha*W_dist + (1-alpha)*W_sim  (default: {ALPHA})"
    )
    parser.add_argument(
        "--sigma2", type=float, default=SIGMA2,
        help=f"Gaussian spread for distance kernel  (default: {SIGMA2})"
    )
    parser.add_argument(
        "--epsilon", type=float, default=EPSILON,
        help=f"Sparsity threshold — edges below this → 0  (default: {EPSILON})"
    )
    parser.add_argument(
        "--T", type=int, default=T_STEPS,
        help=f"Number of synthetic time steps to generate  (default: {T_STEPS})"
    )
    args = parser.parse_args()

    ALPHA   = args.alpha
    SIGMA2  = args.sigma2
    EPSILON = args.epsilon
    T_STEPS = args.T

    main(args.shp, args.db)