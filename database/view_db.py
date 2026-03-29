import duckdb

DB_PATH = "aadhar.duckdb" 
# TABLES  = ["biometric_data", "demographic_data", "enrolment_data"]
TABLES = ["biometric_data_preprocessed", "demographic_data_preprocessed", "enrolment_data_preprocessed"]
con = duckdb.connect(DB_PATH)

for table in TABLES:
    print(f"\n{'='*60}")
    print(f"  TABLE: {table}")
    print(f"{'='*60}")
    df = con.execute(f"SELECT * FROM {table} LIMIT 5").df()
    # df = con.execute(f"SELECT date FROM {table} WHERE CAST(SPLIT_PART(date, '-', 2) AS INT) > 12").df()
    # df = con.execute(f"""
    # SELECT date, YEAR(date), MONTH(date), DAY(date) FROM {table} LIMIT 20""").df()
    print(df.to_string(index=False))

con.close()