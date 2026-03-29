import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("biometric_weekly_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DB_PATH = "../database/aadhar.duckdb"
DISTRICT_ORDER_PATH = "graph_output/district_order.csv"

FEATURE_COLS = [
    "bio_total",
    "bio_age_5_17",
    "bio_age_17_",
    "age_5_ratio",
    "age_17_ratio",
    "dependency_ratio",
]

TARGET_COL = "bio_total"


def load_district_order(path):
    df = pd.read_csv(path)
    if "district" in df.columns:
        return df["district"].tolist()
    return df.iloc[:, 1].tolist()


def fetch_biometric_weekly(con):
    sql = """
    SELECT
        CAST(date_trunc('week', CAST(date AS DATE)) AS DATE) AS week_start,
        district,
        SUM(bio_total) AS bio_total,
        SUM(bio_age_5_17) AS bio_age_5_17,
        SUM(bio_age_17_) AS bio_age_17_,
        AVG(age_5_ratio) AS age_5_ratio,
        AVG(age_17_ratio) AS age_17_ratio,
        AVG(dependency_ratio) AS dependency_ratio
    FROM biometric_data_preprocessed
    WHERE district IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    return con.execute(sql).fetchdf()


def build_full_grid(df, districts):
    df["week_start"] = pd.to_datetime(df["week_start"])
    weeks = pd.date_range(df["week_start"].min(), df["week_start"].max(), freq="W-MON")

    full_index = pd.MultiIndex.from_product(
        [weeks, districts], names=["week_start", "district"]
    )

    df = df.set_index(["week_start", "district"]).reindex(full_index).reset_index()

    count_cols = ["bio_total", "bio_age_5_17", "bio_age_17_"]
    ratio_cols = ["age_5_ratio", "age_17_ratio", "dependency_ratio"]

    for col in count_cols:
        df[col] = df[col].fillna(0.0)

    for col in ratio_cols:
        df[col] = (
            df.groupby("district")[col]
            .transform(lambda s: s.interpolate(limit_direction="both"))
            .fillna(0.0)
        )

    return df, weeks


def build_tensor(df, weeks, districts):
    T = len(weeks)
    N = len(districts)
    C = len(FEATURE_COLS)

    X = np.zeros((T, N, C), dtype=np.float32)
    district_to_idx = {d: i for i, d in enumerate(districts)}
    week_to_idx = {w: i for i, w in enumerate(weeks)}

    for _, row in df.iterrows():
        t = week_to_idx[row["week_start"]]
        n = district_to_idx[row["district"]]
        X[t, n, :] = row[FEATURE_COLS].astype(np.float32).values

    return X


def main():
    districts = load_district_order(DISTRICT_ORDER_PATH)
    con = duckdb.connect(DB_PATH, read_only=True)

    df = fetch_biometric_weekly(con)
    con.close()

    # Keep only districts in graph
    df = df[df["district"].isin(districts)].copy()

    df, weeks = build_full_grid(df, districts)
    X = build_tensor(df, weeks, districts)

    np.save(OUTPUT_DIR / "feature_tensor_X.npy", X)
    pd.Series(FEATURE_COLS, name="feature").to_csv(
        OUTPUT_DIR / "tensor_feature_columns.csv", index=False
    )
    pd.Series(weeks.astype(str), name="week_start").to_csv(
        OUTPUT_DIR / "week_index.csv", index=False
    )

    meta = {
        "target_column": TARGET_COL,
        "shape": X.shape,
        "num_weeks": len(weeks),
        "num_districts": len(districts),
        "num_features": len(FEATURE_COLS),
    }
    pd.Series(meta).to_csv(OUTPUT_DIR / "meta.csv")

    print("Saved biometric weekly tensor:", X.shape)


if __name__ == "__main__":
    main()