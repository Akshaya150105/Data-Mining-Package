import duckdb
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
DB_PATH = "../database/aadhar.duckdb"

BIO_FILE = "biometric_data_preprocessed.csv"
ENROL_FILE ="enrolment_data_preprocessed.csv"


# =========================
# LOAD FUNCTION
# =========================
def load_table(con, file_path, table_name):
    print(f"\nLoading {table_name}...")

    df = pd.read_csv(file_path)

    # Ensure correct types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Create table
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"Loaded {count} rows into {table_name}")


# =========================
# MAIN
# =========================
def main():
    print("Connecting to DuckDB...")
    con = duckdb.connect(DB_PATH)

    # Load both tables
    load_table(con, BIO_FILE, "biometric_data_preprocessed_stgcn")
    load_table(con, ENROL_FILE, "enrolment_data_preprocessed_stgcn")

    # Quick check
    print("\nSample rows:")
    print(con.execute("SELECT * FROM biometric_data_preprocessed_stgcn LIMIT 5").fetchdf())
    print(con.execute("SELECT * FROM enrolment_data_preprocessed_stgcn LIMIT 5").fetchdf())

    con.close()
    print("\nDone. Database ready:", DB_PATH)


if __name__ == "__main__":
    main()