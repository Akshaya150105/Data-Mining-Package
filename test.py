import polars as pl
from rapidfuzz import process, fuzz

# ────────────────────────────────────────────────
#  CONFIG - change only these values if needed
# ────────────────────────────────────────────────
CSV_FILES = [
    r"C:\Users\TEMP.WINSERVER.418\Downloads\api_data_aadhar_biometric_0_500000.csv",
    r"C:\Users\TEMP.WINSERVER.418\Downloads\api_data_aadhar_demographic_0_500000.csv",
    r"C:\Users\TEMP.WINSERVER.418\Downloads\api_data_aadhar_enrolment_0_500000.csv"
]

STANDARD_EXCEL     = r"C:\Users\TEMP.WINSERVER.418\Downloads\LGD - Local Government Directory, Government of India.xlsx"
EXCEL_SHEET        = "Sheet1"
EXCEL_DISTRICT_COL ="District Name (In English)"          

MIN_MATCH_SCORE    = 82
FUZZY_SCORER       = fuzz.WRatio         # alternatives: fuzz.token_sort_ratio / fuzz.token_set_ratio
# ────────────────────────────────────────────────

def load_standard_districts():
    try:
        df_std = pl.read_excel(STANDARD_EXCEL, sheet_name=EXCEL_SHEET)
        districts = (
            df_std
            .select(pl.col(EXCEL_DISTRICT_COL).cast(pl.Utf8).str.strip_chars().str.to_lowercase())
            .unique()
            .to_series()
            .drop_nulls()
            .to_list()
        )
        print(f"Loaded {len(districts)} unique standard district names from {STANDARD_EXCEL}")
        return districts
    except Exception as e:
        print(f"Error reading {STANDARD_EXCEL}: {e}")
        exit(1)


def fuzzy_standardize_district(col: pl.Series, choices: list) -> tuple[pl.Series, list]:
    def match_one(value) -> tuple:
        if not value or str(value).strip() == "":
            return None, None, 0.0

        query = str(value).strip()
        result = process.extractOne(
            query, choices,
            scorer=FUZZY_SCORER,
            score_cutoff=MIN_MATCH_SCORE
        )
        if result is None:
            return query, None, 0.0

        match, score, _ = result
        return match, query, score


    # Return three separate series directly
    matched, orig, sc = zip(*col.map_elements(
        match_one,
        return_dtype=pl.Object   # still Object, but we unpack immediately
    ))

    standardized = pl.Series("matched", matched, dtype=pl.Utf8)
    original     = pl.Series("original", orig,     dtype=pl.Utf8)
    scores       = pl.Series("score",    sc,       dtype=pl.Float64)

    results = col.map_elements(match_one, return_dtype=pl.Object)

    standardized = results.list.get(0)
    original     = results.list.get(1)
    scores       = results.list.get(2)

    changes = []
    for o, s, sc in zip(original, standardized, scores):
        if o is not None and s is not None and o != s:
            changes.append({
                "original": o,
                "standardized": s,
                "score": round(sc, 1)
            })

    return standardized, changes


def process_one_csv(filename: str, std_districts: list):
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    try:
        df = pl.read_csv(filename)
    except Exception as e:
        print(f"→ Cannot read {filename}: {e}")
        return [], 0

    if "district" not in df.columns:
        print("→ No 'district' column → saving without changes")
        out_name = filename.replace(".csv", "_clean.csv")
        df.write_csv(out_name)
        print(f"→ Saved: {out_name}")
        return [], len(df)

    print(f"→ Rows: {len(df):,} | Fuzzy matching 'district'...")

    clean_col, changes = fuzzy_standardize_district(df["district"], std_districts)

    df_clean = df.with_columns(clean_col.alias("district"))

    out_name = filename.replace(".csv", "_clean.csv")
    df_clean.write_csv(out_name)
    print(f"→ Saved cleaned file: {out_name}")

    changed_count = len(changes)
    print(f"→ Districts changed: {changed_count:,d}")

    if 0 < changed_count <= 10:
        print("\nSample changes:")
        for c in changes[:10]:
            print(f"  {c['original']:<28} → {c['standardized']:<28}  (score: {c['score']})")
    elif changed_count > 10:
        print(f"   ... showing first 10 of {changed_count} changes ...")

    return changes, len(df)


def main():
    std_districts = load_standard_districts()

    all_changes = []
    total_rows = 0

    for csv_file in CSV_FILES:
        changes, rows = process_one_csv(csv_file, std_districts)
        all_changes.extend(changes)
        total_rows += rows

    if all_changes:
        log_df = pl.DataFrame(all_changes)
        log_filename = "district_changes_log.csv"
        log_df.write_csv(log_filename)
        print(f"\nFull change log saved → {log_filename}")
        print(f"Total replacements across all files: {len(all_changes):,d}")
    else:
        print("\nNo district names were changed.")

    print(f"\nProcessed {len(CSV_FILES)} files • {total_rows:,d} rows total")
    print("Done.")


if __name__ == "__main__":
    main()