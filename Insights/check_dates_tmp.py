import duckdb

con = duckdb.connect(r"C:\Users\kalya\OneDrive\Documents\Data Mining Package\database\aadhar.duckdb")

tables = ["biometric_data_preprocessed", "demographic_data_preprocessed", "enrolment_data_preprocessed"]
for table in tables:
    print(f"\n--- {table} ---")
    dtype = con.execute(f"SELECT data_type FROM information_schema.columns WHERE table_name='{table}' AND column_name='date'").fetchone()[0]
    nulls = con.execute(f"SELECT COUNT(*) FROM {table} WHERE date IS NULL").fetchone()[0]
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"Data type: {dtype}")
    print(f"Total rows: {total:,}")
    print(f"Null dates: {nulls:,}")
    if nulls > 0:
        print(f"Warning: There are {nulls} rows with NULL dates.")
