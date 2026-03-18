"""
shapefile_prep.py  –  India District Shapefile Preparation for STGCN
=====================================================================
Downloads the GADM 4.1 district-level GeoJSON for India (level 2),
cleans and normalises district/state names to match the Aadhaar DB,
then exports everything STGCN needs:

  Outputs
  -------
  outputs/india_districts.gpkg          – cleaned GeoPackage (geometry + attrs)
  outputs/adjacency_matrix.npy          – (N×N) binary spatial adjacency matrix
  outputs/node_index.csv                – district → node-index mapping
  outputs/centroids.csv                 – district lat/lon centroids
  outputs/districts_by_state.txt        – human-readable district list
  outputs/shapefile_audit.txt           – name-matching audit vs your DB

Why GADM 4.1 over datta07/INDIAN-SHAPEFILES
---------------------------------------------
  • 766 districts, covers post-2019 reorganisation (Ladakh, new TG districts)
  • Consistent title-case names — far less manual cleaning needed
  • Topology is valid (no self-intersections that break adjacency detection)
  • Widely used in published STGCN / spatial-ML papers on India

Usage
-----
    pip install geopandas shapely pyogrio pandas numpy rapidfuzz requests
    python shapefile_prep.py

    # To also match against your DB and see unmatched districts:
    python shapefile_prep.py --db path/to/your.duckdb
"""

import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from shapely.ops import unary_union

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ── GADM 4.1 India level-2 (districts) direct URL ────────────────────────────
# Level 0 = country, Level 1 = states, Level 2 = districts
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json"

# ── GADM column names (level-2 file) ─────────────────────────────────────────
# NAME_1 = state name, NAME_2 = district name
STATE_COL    = "NAME_1"
DISTRICT_COL = "NAME_2"

# ── State name corrections (GADM → canonical Aadhaar DB names) ───────────────
STATE_FIX = {
    "Jammu and Kashmir":                        "Jammu and Kashmir",
    "Dadra and Nagar Haveli and Daman and Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "Andaman and Nicobar":                      "Andaman and Nicobar Islands",
    "Delhi":                                    "Delhi",
    "NCT of Delhi":                             "Delhi",
}

# ── District name corrections (GADM → canonical Aadhaar DB names) ────────────
# Keyed by (state, gadm_district) for safety.  Add more as audit reveals gaps.
DISTRICT_FIX: dict[tuple[str, str], str] = {
    # Telangana
    ("Telangana", "Medchal Malkajgiri"):   "Medchal-Malkajgiri",
    ("Telangana", "Yadadri Bhuvanagiri"):  "Yadadri-Bhuvanagiri",
    ("Telangana", "Bhadradri Kothagudem"):"Bhadradri Kothagudem",
    # Maharashtra — renames
    ("Maharashtra", "Aurangabad"):         "Chhatrapati Sambhajinagar",
    ("Maharashtra", "Osmanabad"):          "Dharashiv",
    # Madhya Pradesh — renames
    ("Madhya Pradesh", "Hoshangabad"):     "Narmadapuram",
    # Karnataka
    ("Karnataka", "Bangalore"):            "Bengaluru Urban",
    ("Karnataka", "Bangalore Rural"):      "Bengaluru Rural",
    ("Karnataka", "Mysore"):               "Mysuru",
    ("Karnataka", "Shimoga"):              "Shivamogga",
    ("Karnataka", "Tumkur"):               "Tumakuru",
    ("Karnataka", "Gulbarga"):             "Kalaburagi",
    ("Karnataka", "Belgaum"):              "Belagavi",
    ("Karnataka", "Bijapur"):              "Vijayapura",
    ("Karnataka", "Bellary"):              "Ballari",
    ("Karnataka", "Davangere"):            "Davanagere",
    ("Karnataka", "Chikmagalur"):          "Chikkamagaluru",
    ("Karnataka", "Chamrajnagar"):         "Chamarajanagar",
    ("Karnataka", "Ramanagar"):            "Ramanagara",
    ("Karnataka", "Chikkaballapur"):       "Chikkaballapura",
    # Uttar Pradesh — renames
    ("Uttar Pradesh", "Allahabad"):        "Prayagraj",
    ("Uttar Pradesh", "Faizabad"):         "Ayodhya",
    # Odisha
    ("Odisha", "Anugul"):                  "Anugul",
    ("Odisha", "Jagatsinghapur"):          "Jagatsinghapur",
    ("Odisha", "Baudh"):                   "Boudh",
    ("Odisha", "Subarnapur"):              "Subarnapur",
    # West Bengal
    ("West Bengal", "Barddhaman"):         "Paschim Bardhaman",
    ("West Bengal", "Darjiling"):          "Darjeeling",
    ("West Bengal", "Haora"):              "Howrah",
    ("West Bengal", "Hugli"):              "Hooghly",
    ("West Bengal", "Koch Bihar"):         "Cooch Behar",
    ("West Bengal", "Maldah"):             "Malda",
    ("West Bengal", "Puruliya"):           "Purulia",
    ("West Bengal", "Uttar Dinajpur"):     "North Dinajpur",
    ("West Bengal", "Dakshin Dinajpur"):   "South Dinajpur",
    ("West Bengal", "Medinipur"):          "Paschim Medinipur",
    # Punjab
    ("Punjab", "Firozpur"):                "Ferozepur",
    ("Punjab", "Muktsar"):                 "Sri Muktsar Sahib",
    ("Punjab", "Nawanshahr"):              "Shahid Bhagat Singh Nagar",
    # Bihar
    ("Bihar", "Purnia"):                   "Purnea",
    ("Bihar", "Monghyr"):                  "Munger",
    ("Bihar", "Samstipur"):                "Samastipur",
    # Rajasthan
    ("Rajasthan", "Dhaulpur"):             "Dholpur",
    ("Rajasthan", "Jalor"):                "Jalore",
    ("Rajasthan", "Jhunjhunun"):           "Jhunjhunu",
    ("Rajasthan", "Chittaurgarh"):         "Chittorgarh",
    # Gujarat
    ("Gujarat", "Panch Mahals"):           "Panchmahal",
    ("Gujarat", "Banas Kantha"):           "Banaskantha",
    ("Gujarat", "Sabar Kantha"):           "Sabarkantha",
    ("Gujarat", "Dohad"):                  "Dahod",
    # Tamil Nadu
    ("Tamil Nadu", "Kancheepuram"):        "Kanchipuram",
    ("Tamil Nadu", "Kanniyakumari"):       "Kanyakumari",
    ("Tamil Nadu", "Tuticorin"):           "Thoothukudi",
    ("Tamil Nadu", "Villupuram"):          "Viluppuram",
    ("Tamil Nadu", "Thiruvarur"):          "Tiruvarur",
    ("Tamil Nadu", "Thiruvallur"):         "Tiruvallur",
    # Andhra Pradesh
    ("Andhra Pradesh", "Cuddapah"):        "Y S R",
    ("Andhra Pradesh", "Anantapur"):       "Ananthapuramu",
    ("Andhra Pradesh", "Visakhapatanam"): "Visakhapatnam",
    # Himachal Pradesh
    ("Himachal Pradesh", "Lahul And Spiti"): "Lahaul And Spiti",
    # Jammu and Kashmir
    ("Jammu and Kashmir", "Badgam"):       "Budgam",
    ("Jammu and Kashmir", "Rajauri"):      "Rajouri",
    ("Jammu and Kashmir", "Poonch"):       "Punch",
    ("Jammu and Kashmir", "Shupiyan"):     "Shopian",
    ("Jammu and Kashmir", "Bandipore"):    "Bandipur",
    # Jharkhand
    ("Jharkhand", "Pakaur"):               "Pakur",
    ("Jharkhand", "Sahebganj"):            "Sahibganj",
    ("Jharkhand", "Koderma"):              "Kodarma",
    ("Jharkhand", "Palamau"):              "Palamu",
    # Assam
    ("Assam", "North Cachar Hills"):       "Dima Hasao",
    ("Assam", "Sibsagar"):                 "Sivasagar",
    # Kerala
    ("Kerala", "Kasaragod"):               "Kasaragod",
    # Haryana
    ("Haryana", "Gurgaon"):                "Gurugram",
    ("Haryana", "Mewat"):                  "Nuh",
    # Chhattisgarh
    ("Chhattisgarh", "Kawardha"):          "Kabirdham",
    ("Chhattisgarh", "Uttar Bastar Kanker"): "Kanker",
    ("Chhattisgarh", "Dakshin Bastar Dantewada"): "Dantewada",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fix_encoding(s: str) -> str:
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[\u2013\u2014\u2212]", "-", s)


def _clean(s: str) -> str:
    if not s:
        return ""
    s = _fix_encoding(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ── STEP 1: Load GADM 4.1 ─────────────────────────────────────────────────────
def load_gadm() -> gpd.GeoDataFrame:
    print("=" * 60)
    print("STEP 1 – Loading GADM 4.1 India districts")
    print("=" * 60)

    # Cache locally so re-runs are fast
    cache = OUT_DIR / "gadm41_IND_2.json"
    if cache.exists():
        print(f"  Using cached file: {cache}")
        gdf = gpd.read_file(cache)
    else:
        print(f"  Downloading from:\n  {GADM_URL}")
        gdf = gpd.read_file(GADM_URL)
        gdf.to_file(cache, driver="GeoJSON")
        print(f"  Cached → {cache}")

    print(f"  Loaded {len(gdf)} features")
    print(f"  CRS: {gdf.crs}")
    print(f"  Columns: {gdf.columns.tolist()}")
    return gdf


# ── STEP 2: Clean and normalise names ────────────────────────────────────────
def clean_names(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("\n" + "=" * 60)
    print("STEP 2 – Normalising state and district names")
    print("=" * 60)

    gdf = gdf.copy()

    # Keep only needed columns + geometry
    gdf = gdf[[STATE_COL, DISTRICT_COL, "geometry"]].copy()
    gdf = gdf.rename(columns={STATE_COL: "state_raw", DISTRICT_COL: "district_raw"})

    # Fix state names
    gdf["state"] = gdf["state_raw"].map(lambda x: STATE_FIX.get(x, x))

    # Apply district fixes (state-scoped)
    def fix_district(row):
        key = (row["state"], row["district_raw"])
        return DISTRICT_FIX.get(key, row["district_raw"])

    gdf["district"] = gdf.apply(fix_district, axis=1)

    # Validate geometry — fix any invalid polygons
    invalid = (~gdf.geometry.is_valid).sum()
    if invalid:
        print(f"  Fixing {invalid} invalid geometries …")
        gdf["geometry"] = gdf.geometry.buffer(0)

    # Ensure WGS84
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    print(f"  States/UTs   : {gdf['state'].nunique()}")
    print(f"  Districts    : {len(gdf)}")
    print(f"  Unique dist  : {gdf['district'].nunique()}")

    # Warn if any district name is duplicated within a state (shouldn't happen)
    dups = gdf.groupby(["state", "district"]).size()
    dups = dups[dups > 1]
    if not dups.empty:
        print(f"\n  WARNING: {len(dups)} duplicate (state, district) pairs:")
        for (s, d), n in dups.items():
            print(f"    [{s}] {d}  ×{n}")

    return gdf


# ── STEP 3: Build adjacency matrix ───────────────────────────────────────────
def build_adjacency(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build a binary N×N spatial adjacency matrix.
    Two districts are adjacent if their geometries share a border
    (touch or overlap — using a small buffer to handle floating-point gaps).

    Returns
    -------
    adj   : np.ndarray  shape (N, N), dtype uint8
    index : pd.DataFrame  columns [node_id, state, district]
    """
    print("\n" + "=" * 60)
    print("STEP 3 – Building spatial adjacency matrix")
    print("=" * 60)

    gdf = gdf.reset_index(drop=True)
    N   = len(gdf)

    # Node index table
    index = gdf[["state", "district"]].copy()
    index.insert(0, "node_id", range(N))

    # Use a tiny buffer (0.01 degrees ≈ 1 km) to bridge micro-gaps in polygons
    BUFFER = 0.01
    geoms_buffered = gdf.geometry.buffer(BUFFER)

    adj = np.zeros((N, N), dtype=np.uint8)

    print(f"  Building {N}×{N} adjacency matrix …")
    for i in range(N):
        for j in range(i + 1, N):
            if geoms_buffered.iloc[i].intersects(geoms_buffered.iloc[j]):
                adj[i, j] = 1
                adj[j, i] = 1

    # Self-loops off (STGCN adds them internally)
    np.fill_diagonal(adj, 0)

    n_edges = adj.sum() // 2
    avg_deg = adj.sum(axis=1).mean()
    print(f"  Edges (unique borders) : {n_edges}")
    print(f"  Avg degree per district: {avg_deg:.1f}")

    # Sanity: any isolated nodes?
    isolated = (adj.sum(axis=1) == 0).sum()
    if isolated:
        print(f"  WARNING: {isolated} district(s) have no neighbours — check geometry")
        iso_names = index.loc[adj.sum(axis=1) == 0, ["state", "district"]]
        print(iso_names.to_string(index=False))

    return adj, index


# ── STEP 4: Compute centroids ─────────────────────────────────────────────────
def compute_centroids(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 4 – Computing district centroids")
    print("=" * 60)

    gdf = gdf.reset_index(drop=True)
    centroids = gdf.geometry.centroid
    df = pd.DataFrame({
        "node_id":  range(len(gdf)),
        "state":    gdf["state"],
        "district": gdf["district"],
        "lat":      centroids.y,
        "lon":      centroids.x,
    })
    print(f"  Computed {len(df)} centroids.")
    return df


# ── STEP 5: DB audit (optional) ───────────────────────────────────────────────
def audit_vs_db(gdf: gpd.GeoDataFrame, db_path: str) -> None:
    print("\n" + "=" * 60)
    print("STEP 5 – Auditing shapefile names vs database")
    print("=" * 60)
    try:
        import duckdb
    except ImportError:
        print("  duckdb not installed — skipping DB audit.")
        return

    con = duckdb.connect(db_path)

    db_pairs = set()
    for tbl in ["biometric_data", "demographic_data", "enrolment_data"]:
        try:
            rows = con.execute(
                f"SELECT DISTINCT state, district FROM {tbl}"
            ).fetchall()
            db_pairs.update(rows)
        except Exception as e:
            print(f"  Could not query {tbl}: {e}")
    con.close()

    shp_pairs = set(zip(gdf["state"], gdf["district"]))

    in_db_not_shp  = db_pairs  - shp_pairs
    in_shp_not_db  = shp_pairs - db_pairs

    audit_path = OUT_DIR / "shapefile_audit.txt"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("Shapefile ↔ Database Audit\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Shapefile districts : {len(shp_pairs)}\n")
        f.write(f"DB districts        : {len(db_pairs)}\n")
        f.write(f"Matched             : {len(shp_pairs & db_pairs)}\n\n")

        f.write(f"In DB but NOT in shapefile ({len(in_db_not_shp)}):\n")
        f.write("-" * 50 + "\n")
        for s, d in sorted(in_db_not_shp):
            f.write(f"  [{s}] {d}\n")

        f.write(f"\nIn shapefile but NOT in DB ({len(in_shp_not_db)}):\n")
        f.write("-" * 50 + "\n")
        for s, d in sorted(in_shp_not_db):
            f.write(f"  [{s}] {d}\n")

    print(f"  Matched  : {len(shp_pairs & db_pairs)}")
    print(f"  In DB, missing from shapefile : {len(in_db_not_shp)}")
    print(f"  In shapefile, missing from DB : {len(in_shp_not_db)}")
    print(f"  Audit written → {audit_path}")


# ── STEP 6: Save outputs ──────────────────────────────────────────────────────
def save_outputs(
    gdf:      gpd.GeoDataFrame,
    adj:      np.ndarray,
    index:    pd.DataFrame,
    centroids: pd.DataFrame,
) -> None:
    print("\n" + "=" * 60)
    print("STEP 6 – Saving outputs")
    print("=" * 60)

    # GeoPackage (geometry + clean attrs)
    gpkg_path = OUT_DIR / "india_districts.gpkg"
    gdf[["state", "district", "geometry"]].to_file(gpkg_path, driver="GPKG")
    print(f"  GeoPackage   → {gpkg_path}")

    # Adjacency matrix
    adj_path = OUT_DIR / "adjacency_matrix.npy"
    np.save(adj_path, adj)
    print(f"  Adjacency    → {adj_path}  shape={adj.shape}")

    # Also save as CSV for inspection
    adj_csv = OUT_DIR / "adjacency_matrix.csv"
    pd.DataFrame(adj,
                 index=index["district"],
                 columns=index["district"]).to_csv(adj_csv)
    print(f"  Adjacency CSV→ {adj_csv}")

    # Node index
    idx_path = OUT_DIR / "node_index.csv"
    index.to_csv(idx_path, index=False)
    print(f"  Node index   → {idx_path}")

    # Centroids
    cen_path = OUT_DIR / "centroids.csv"
    centroids.to_csv(cen_path, index=False)
    print(f"  Centroids    → {cen_path}")

    # Human-readable district list
    txt_path = OUT_DIR / "districts_by_state.txt"
    districts_by_state = (
        gdf.groupby("state")["district"]
        .apply(lambda x: sorted(x.tolist()))
        .to_dict()
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Districts by State (GADM 4.1, cleaned)\n")
        f.write("=" * 50 + "\n\n")
        for state in sorted(districts_by_state):
            dlist = districts_by_state[state]
            f.write(f"{state} ({len(dlist)} districts):\n")
            f.write("-" * (len(state) + 15) + "\n")
            for d in dlist:
                f.write(f"  - {d}\n")
            f.write("\n")
    print(f"  District list→ {txt_path}")


# ── Summary stats (mirrors original script) ───────────────────────────────────
def print_summary(gdf: gpd.GeoDataFrame) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    state_counts = (
        gdf.groupby("state")["district"]
        .count()
        .reset_index()
        .rename(columns={"district": "district_count"})
        .sort_values("district_count", ascending=False)
        .reset_index(drop=True)
    )
    state_counts.index += 1

    print(f"\n{'Rank':<5} {'State/UT':<40} {'Districts':>9}")
    print("-" * 58)
    for rank, row in state_counts.iterrows():
        print(f"{rank:<5} {row['state']:<40} {row['district_count']:>9}")
    print("-" * 58)
    print(f"{'TOTAL':<46} {state_counts['district_count'].sum():>9}\n")

    print(f"Avg districts per state : {state_counts['district_count'].mean():.1f}")
    print(f"Max  : {state_counts['district_count'].max()}  ({state_counts.iloc[0]['state']})")
    print(f"Min  : {state_counts['district_count'].min()}  ({state_counts.iloc[-1]['state']})")

    print("\nDistrict count buckets:")
    bins   = [0, 5, 10, 20, 30, 50, 100]
    labels = ["1–5", "6–10", "11–20", "21–30", "31–50", "51+"]
    state_counts["bucket"] = pd.cut(
        state_counts["district_count"], bins=bins, labels=labels
    )
    print(state_counts.groupby("bucket", observed=True)["state"].count().to_string())

    print("\nMissing values:")
    print(gdf[["state", "district"]].isnull().sum().to_string())

    print("\nSample rows:")
    print(gdf[["state", "district"]].sample(5).to_string())


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="India District Shapefile Preparation for STGCN"
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to your DuckDB database file for name-matching audit (optional)"
    )
    parser.add_argument(
        "--skip-adjacency", action="store_true",
        help="Skip the adjacency matrix build (slow for 700+ districts)"
    )
    args = parser.parse_args()

    gdf      = load_gadm()
    gdf      = clean_names(gdf)
    centroids = compute_centroids(gdf)

    if not args.skip_adjacency:
        adj, index = build_adjacency(gdf)
    else:
        print("\nSkipping adjacency matrix (--skip-adjacency set).")
        index = gdf[["state", "district"]].copy().reset_index(drop=True)
        index.insert(0, "node_id", range(len(gdf)))
        adj   = np.zeros((len(gdf), len(gdf)), dtype=np.uint8)

    if args.db:
        audit_vs_db(gdf, args.db)

    save_outputs(gdf, adj, index, centroids)
    print_summary(gdf)

    print("\n✓ Done. All outputs in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()