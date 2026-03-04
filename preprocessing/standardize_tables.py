import pandas as pd
from utils.db_connection import get_connection
from preprocessing.district_cleaning import cluster_similar_names

con = get_connection()

tables = [
    "biometric_data",
    "demographic_data",
    "enrolment_data"
]

for table in tables:

    df = con.execute(f"SELECT * FROM {table}").df()

    df["district"] = df["district"].str.lower().str.strip()

    districts = df["district"].unique()

    clusters = cluster_similar_names(districts)

    for group in clusters:

        canonical = group[0]

        for variant in group:
            df.loc[df["district"] == variant, "district"] = canonical

    con.execute(f"CREATE OR REPLACE TABLE {table}_clean AS SELECT * FROM df")

    print(f"{table} cleaned")