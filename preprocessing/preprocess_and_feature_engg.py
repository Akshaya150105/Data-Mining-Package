import duckdb
import pandas as pd
import numpy as np
from typing import Dict

DB_PATH = "../database/aadhar.duckdb"
TABLES  = ["biometric_data", "demographic_data", "enrolment_data"]

con = duckdb.connect(DB_PATH)

# ==============================================================
# CLEAN FUNCTIONS (state/district names are never touched)
# ==============================================================

def validate_data(df: pd.DataFrame) -> Dict:
    report = {
        "total_rows":      len(df),
        "total_columns":   len(df.columns),
        "columns":         df.columns.tolist(),
        "dtypes":          df.dtypes.astype(str).to_dict(),
        "missing_values":  df.isnull().sum().to_dict(),
        "duplicate_rows":  int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
    }
    return report

def convert_date_column(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    df[column] = pd.to_datetime(df[column], errors="coerce")
    invalid = df[column].isna().sum()
    if invalid > 0:
        print(f"  ⚠ {invalid} invalid dates converted to NaT")
    return df

def handle_duplicates(df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
    dup_count = df.duplicated().sum()
    print(f"  Duplicate rows found: {dup_count}")
    if drop and dup_count > 0:
        df = df.drop_duplicates()
        print(f"  Duplicates removed. New shape: {df.shape}")
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = "fill") -> pd.DataFrame:
    # Protect state and district from being altered
    protected = [c for c in ["state", "district"] if c in df.columns]

    missing_total = df.isnull().sum().sum()
    print(f"  Total missing values: {missing_total}")
    if missing_total == 0:
        return df

    if strategy == "drop":
        df = df.dropna()
    elif strategy == "fill":
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)

        # Fill object cols EXCEPT state and district
        cat_cols = [c for c in df.select_dtypes(include=["object"]).columns
                    if c not in protected]
        df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df

def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  --- Date Completeness Check ---")
    min_date     = df["date"].min()
    max_date     = df["date"].max()
    unique_dates = df["date"].nunique()
    expected     = (max_date - min_date).days + 1
    print(f"  Date range   : {min_date} → {max_date}")
    print(f"  Unique dates : {unique_dates} / {expected} expected")
    print(f"  Missing dates: {expected - unique_dates}")
    return df

# ==============================================================
# FEATURE ENGINEERING (only applies to biometric_data columns)
# ==============================================================

def create_biometric_features(df: pd.DataFrame) -> pd.DataFrame:
    if not {"bio_age_5_17", "bio_age_17_"}.issubset(df.columns):
        return df  # skip for tables without these columns
    df["bio_total"]         = df["bio_age_5_17"] + df["bio_age_17_"]
    df["age_5_ratio"]       = df["bio_age_5_17"] / (df["bio_total"] + 1)
    df["age_17_ratio"]      = df["bio_age_17_"]  / (df["bio_total"] + 1)
    df["dominant_age_group"]= np.where(df["bio_age_5_17"] > df["bio_age_17_"], "0-5", "5-17")
    df["dependency_ratio"]  = df["bio_age_5_17"] / (df["bio_age_17_"] + 1)
    return df

def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    if not {"demo_age_5_17", "demo_age_17_"}.issubset(df.columns):
        return df
    df["demo_total"]             = df["demo_age_5_17"] + df["demo_age_17_"]
    df["demo_age_5_ratio"]       = df["demo_age_5_17"] / (df["demo_total"] + 1)
    df["demo_age_17_ratio"]      = df["demo_age_17_"]  / (df["demo_total"] + 1)
    df["demo_dependency_ratio"]  = df["demo_age_5_17"] / (df["demo_age_17_"] + 1)
    return df

def create_enrolment_features(df: pd.DataFrame) -> pd.DataFrame:
    if not {"age_0_5", "age_5_17", "age_18_greater"}.issubset(df.columns):
        return df
    df["enrol_total"]        = df["age_0_5"] + df["age_5_17"] + df["age_18_greater"]
    df["enrol_minor_ratio"]  = (df["age_0_5"] + df["age_5_17"]) / (df["enrol_total"] + 1)
    df["enrol_adult_ratio"]  = df["age_18_greater"] / (df["enrol_total"] + 1)
    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["month_name"]  = df["date"].dt.strftime("%B")
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"]  = df["date"].dt.dayofweek >= 5
    df["quarter"]     = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear
    return df

def create_aggregated_features(df: pd.DataFrame, total_col: str) -> pd.DataFrame:
    if total_col not in df.columns:
        return df
    state_total = (df.groupby("state")[total_col]
                     .sum().reset_index(name="state_total"))
    df = df.merge(state_total, on="state", how="left")

    district_total = (df.groupby(["state", "district"])[total_col]
                        .sum().reset_index(name="district_total"))
    df = df.merge(district_total, on=["state", "district"], how="left")
    return df

def create_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    if "district_total" not in df.columns:
        return df
    df["district_rank_in_state"] = (
        df.groupby("state")["district_total"]
          .rank(method="dense", ascending=False)
    )
    return df

def create_growth_features(df: pd.DataFrame, total_col: str) -> pd.DataFrame:
    if total_col not in df.columns:
        return df
    df = df.sort_values(["state", "district", "pincode", "date"])
    grp = df.groupby(["state", "district", "pincode"])[total_col]

    df["daily_change"]     = grp.diff().fillna(0)
    df["daily_pct_change"] = (grp.pct_change() * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    df[f"{total_col}_7day_avg"] = (
        grp.rolling(window=7, min_periods=1).mean()
           .reset_index(level=[0,1,2], drop=True)
           .fillna(df[total_col])
    )
    df[f"{total_col}_7day_std"] = (
        grp.rolling(window=7, min_periods=1).std()
           .reset_index(level=[0,1,2], drop=True)
           .fillna(0)
    )
    return df

# ==============================================================
# TABLE-SPECIFIC TOTAL COLUMN MAPPING
# ==============================================================
TOTAL_COL_MAP = {
    "biometric_data":    "bio_total",
    "demographic_data":  "demo_total",
    "enrolment_data":    "enrol_total",
}

# ==============================================================
# MAIN PIPELINE
# ==============================================================

for table in TABLES:
    print(f"\n{'='*60}")
    print(f"  Processing: {table}")
    print(f"{'='*60}")

    df = con.execute(f"SELECT * FROM {table}").df()

    # --- Validate ---
    report = validate_data(df)
    print(f"  Rows: {report['total_rows']} | Cols: {report['total_columns']}")
    print(f"  Missing: {report['missing_values']} | Duplicates: {report['duplicate_rows']}")

    # --- Clean ---
    df = convert_date_column(df)
    df = handle_duplicates(df)
    df = handle_missing_values(df, strategy="fill")
    df = fill_missing_dates(df)

    # --- Feature Engineering ---
    print("\n  --- Feature Engineering ---")
    df = create_biometric_features(df)
    df = create_demographic_features(df)
    df = create_enrolment_features(df)
    df = create_time_features(df)

    total_col = TOTAL_COL_MAP.get(table)
    df = create_aggregated_features(df, total_col)
    df = create_rank_features(df)
    df = create_growth_features(df, total_col)

    print(f"  ✓ Features done | Total columns: {len(df.columns)}")
    print(f"  Remaining NaNs : {df.isnull().sum().sum()}")

    # --- Save to DuckDB ---
    out_table = f"{table}_preprocessed"
    con.execute(f"DROP TABLE IF EXISTS {out_table}")
    con.execute(f"CREATE TABLE {out_table} AS SELECT * FROM df")
    print(f"  ✓ Saved → {out_table}")

con.close()
print("\n✅ All tables preprocessed and saved.")