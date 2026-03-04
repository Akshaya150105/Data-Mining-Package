import glob
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_connection import get_connection


con = get_connection()

con.execute("""
CREATE OR REPLACE TABLE biometric_data (
    id           UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
    date         DATE         NOT NULL,
    state        VARCHAR      NOT NULL,
    district     VARCHAR      NOT NULL,
    pincode      VARCHAR(6)   NOT NULL,
    bio_age_5_17 INTEGER      NOT NULL CHECK (bio_age_5_17 >= 0),
    bio_age_17_  INTEGER      NOT NULL CHECK (bio_age_17_ >= 0)
);
""")

con.execute("""
CREATE OR REPLACE TABLE demographic_data (
    id            UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
    date          DATE         NOT NULL,
    state         VARCHAR      NOT NULL,
    district      VARCHAR      NOT NULL,
    pincode       VARCHAR(6)   NOT NULL CHECK (length(pincode) = 6),
    demo_age_5_17 INTEGER      NOT NULL CHECK (demo_age_5_17 >= 0),
    demo_age_17_  INTEGER      NOT NULL CHECK (demo_age_17_ >= 0)
);
""")

con.execute("""
CREATE OR REPLACE TABLE enrolment_data (
    id             UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
    date           DATE         NOT NULL,
    state          VARCHAR      NOT NULL,
    district       VARCHAR      NOT NULL,
    pincode        VARCHAR(6)   NOT NULL CHECK (length(pincode) = 6),
    age_0_5        INTEGER      NOT NULL CHECK (age_0_5 >= 0),
    age_5_17       INTEGER      NOT NULL CHECK (age_5_17 >= 0),
    age_18_greater INTEGER      NOT NULL CHECK (age_18_greater >= 0)
);
""")


def load_table(table_name, csv_pattern, columns):
    print(table_name)
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        print("No CSV files found in the directory!")
        print("Checked path:", csv_pattern)
        return

    print(f"Found {len(csv_files)} CSV files:")
    for file_path in csv_files:
        print("  ", file_path)

    for csv_path in csv_files:
        print(f"\nLoading: {csv_path}")
        before_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        try:
            con.execute(
                f"""
                COPY {table_name} ({columns})
                FROM '{csv_path}'
                (DELIMITER ',', HEADER TRUE, quote '"', escape '"', nullstr '');
                """
            )
            after_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            inserted = after_count - before_count
            print(f"  -> inserted {inserted:,} rows")
        except Exception as exc:
            print(f"  ERROR loading {csv_path}:")
            print(" ", str(exc).encode("ascii", "replace").decode("ascii"))

    total_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"\nTotal rows loaded: {total_rows:,}")


load_table(
    "biometric_data",
    "../Data/api_data_aadhar_biometric/*.csv",
    "date, state, district, pincode, bio_age_5_17, bio_age_17_",
)

load_table(
    "demographic_data",
    "../Data/api_data_aadhar_demographic/*.csv",
    "date, state, district, pincode, demo_age_5_17, demo_age_17_",
)

load_table(
    "enrolment_data",
    "../Data/api_data_aadhar_enrolment/*.csv",
    "date, state, district, pincode, age_0_5, age_5_17, age_18_greater",
)

print("Biometric\nSample rows:")
print(con.sql("SELECT * FROM biometric_data LIMIT 5").fetchall())
print("\nRow count:", con.sql("SELECT COUNT(*) FROM biometric_data").fetchone()[0])

print("Demographic\nSample rows:")
print(con.sql("SELECT * FROM demographic_data LIMIT 5").fetchall())
print("\nRow count:", con.sql("SELECT COUNT(*) FROM demographic_data").fetchone()[0])

print("Enrolment\nSample rows:")
print(con.sql("SELECT * FROM enrolment_data LIMIT 5").fetchall())
print("\nRow count:", con.sql("SELECT COUNT(*) FROM enrolment_data").fetchone()[0])

con.close()
