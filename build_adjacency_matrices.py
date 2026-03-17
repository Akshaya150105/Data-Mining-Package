"""
build_adjacency_matrices.py
============================
Builds 3 binary adjacency matrices (one per DB table) for Indian districts,
using spatial data from 2011_Dist.shp and district names from the preprocessed
DuckDB tables.

Outputs (written to ./adjacency_output/):
  - biometric_data_adjacency.csv
  - demographic_data_adjacency.csv
  - enrolment_data_adjacency.csv
  - biometric_data_graph.png
  - demographic_data_graph.png
  - enrolment_data_graph.png
  - unmatched_districts.log   (DB districts that had no shapefile match)

Usage
-----
    python build_adjacency_matrices.py --shp path/to/2011_Dist.shp

Dependencies
------------
    pip install geopandas shapely numpy pandas networkx matplotlib rapidfuzz duckdb
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_connection import get_connection

# ── Config ─────────────────────────────────────────────────────────────────────
TABLES        = ["biometric_data", "demographic_data", "enrolment_data"]
FUZZY_THRESH  = 85          # RapidFuzz score threshold for name matching
OUTPUT_DIR    = Path(__file__).parent / "adjacency_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
RUN_TS   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = OUTPUT_DIR / f"unmatched_districts_{RUN_TS}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────
def normalise(name: str) -> str:
    """Lowercase + strip + collapse whitespace."""
    if not isinstance(name, str):
        return ""
    return " ".join(name.lower().split())


def match_districts(
    db_districts: list[str],
    shp_districts: list[str],
    table_name: str,
) -> tuple[dict[str, str], list[str]]:
    """
    Fuzzy-match each DB district name against the shapefile district list.
    Returns:
        matched   : {db_name -> shp_name}  (only successfully matched)
        unmatched : [db_name, ...]          (no match above threshold)
    """
    shp_norm = {normalise(s): s for s in shp_districts}
    matched:   dict[str, str] = {}
    unmatched: list[str]      = []

    for db_name in db_districts:
        norm = normalise(db_name)
        res  = process.extractOne(norm, list(shp_norm.keys()), scorer=fuzz.WRatio)
        if res and res[1] >= FUZZY_THRESH:
            matched[db_name] = shp_norm[res[0]]
        else:
            unmatched.append(db_name)

    log.info(
        "  [%s] Matched %d / %d DB districts to shapefile.",
        table_name, len(matched), len(db_districts),
    )
    if unmatched:
        log.warning(
            "  [%s] %d unmatched district(s) — logged and skipped:\n    %s",
            table_name, len(unmatched), "\n    ".join(unmatched),
        )
    return matched, unmatched


def build_adjacency(
    gdf: gpd.GeoDataFrame,
    district_col: str,
    districts_subset: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Build a binary N×N adjacency matrix for *districts_subset* using
    spatial touches() on the shapefile geometries.

    Returns (A, ordered_district_list).
    """
    # Filter GDF to only the districts present in DB for this table
    mask = gdf[district_col].isin(districts_subset)
    sub  = gdf[mask].reset_index(drop=True)

    # Deduplicate: some shapefiles have duplicate names (e.g. split polygons)
    # Dissolve so each district name becomes one (multi)polygon
    sub = sub.dissolve(by=district_col).reset_index()

    districts = sub[district_col].tolist()
    N         = len(districts)
    A         = np.zeros((N, N), dtype=np.int8)

    log.info("  Building %d×%d adjacency matrix ...", N, N)

    for i, j in combinations(range(N), 2):
        if sub.geometry[i].touches(sub.geometry[j]):
            A[i, j] = 1
            A[j, i] = 1

    return A, districts


def save_matrix_csv(A: np.ndarray, districts: list[str], path: Path) -> None:
    df = pd.DataFrame(A, index=districts, columns=districts)
    df.to_csv(path)
    log.info("  Saved matrix → %s", path)


def save_graph_png(
    A: np.ndarray,
    districts: list[str],
    table_name: str,
    path: Path,
) -> None:
    G = nx.from_numpy_array(A)
    mapping = {i: d for i, d in enumerate(districts)}
    G = nx.relabel_nodes(G, mapping)

    fig, ax = plt.subplots(figsize=(22, 18))

    # Use spring layout — decent for geographic adjacency graphs
    pos = nx.spring_layout(G, seed=42, k=1.5 / (len(districts) ** 0.5))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60,
                           node_color="#4A90D9", alpha=0.85)
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.5,
                           edge_color="#AAAAAA", alpha=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=4,
                            font_color="#1a1a1a")

    ax.set_title(
        f"District Adjacency Graph — {table_name}\n"
        f"({len(districts)} districts, {G.number_of_edges()} edges)",
        fontsize=14, fontweight="bold", pad=16,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved graph PNG → %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(shp_path: str) -> None:
    log.info("=" * 60)
    log.info("Loading shapefile: %s", shp_path)
    log.info("=" * 60)

    gdf = gpd.read_file(shp_path)

    # Detect district column (notebook used "DISTRICT")
    dist_col = None
    for candidate in ["DISTRICT", "district", "District", "DISTNAME", "dtname"]:
        if candidate in gdf.columns:
            dist_col = candidate
            break
    if dist_col is None:
        raise ValueError(
            f"Could not find a district column in shapefile. "
            f"Available columns: {list(gdf.columns)}"
        )

    log.info("Shapefile district column: '%s'  |  %d rows", dist_col, len(gdf))

    # Normalise shapefile district names to Title Case for consistent matching
    gdf[dist_col] = gdf[dist_col].str.strip().str.title()
    shp_districts = gdf[dist_col].dropna().unique().tolist()
    log.info("Unique districts in shapefile: %d", len(shp_districts))

    # ── Connect to DB ──────────────────────────────────────────────────────────
    con = get_connection()

    all_unmatched: dict[str, list[str]] = {}

    for table in TABLES:
        log.info("")
        log.info("─" * 50)
        log.info("Processing table: %s", table)
        log.info("─" * 50)

        # Pull distinct (state, district) from DB
        rows = con.execute(
            f"SELECT DISTINCT district FROM {table} WHERE district IS NOT NULL"
        ).fetchall()
        db_districts = [r[0] for r in rows]
        log.info("  Distinct DB districts: %d", len(db_districts))

        # Match DB names → shapefile names
        matched, unmatched = match_districts(db_districts, shp_districts, table)
        all_unmatched[table] = unmatched

        if not matched:
            log.error("  No districts matched for %s — skipping.", table)
            continue

        # The subset of shapefile district names we care about
        shp_subset = list(set(matched.values()))

        # Build adjacency matrix
        A, ordered = build_adjacency(gdf, dist_col, shp_subset)

        # Save CSV
        csv_path = OUTPUT_DIR / f"{table}_adjacency.csv"
        save_matrix_csv(A, ordered, csv_path)

        # Save graph PNG
        png_path = OUTPUT_DIR / f"{table}_graph.png"
        save_graph_png(A, ordered, table, png_path)

    con.close()

    # ── Write consolidated unmatched log ───────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Unmatched district summary")
    log.info("=" * 60)
    for tbl, names in all_unmatched.items():
        if names:
            log.warning("  %s (%d unmatched): %s", tbl, len(names), names)
        else:
            log.info("  %s: all districts matched ✓", tbl)

    log.info("")
    log.info("All outputs written to: %s", OUTPUT_DIR)
    log.info("Unmatched log: %s", LOG_PATH)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build binary district adjacency matrices from DuckDB + shapefile."
    )
    parser.add_argument(
        "--shp",
        required=True,
        help="Adjacency_marix\2011_Dist.shp",
    )
    args = parser.parse_args()
    main(args.shp)