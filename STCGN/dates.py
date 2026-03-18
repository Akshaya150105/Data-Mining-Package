import duckdb
con = duckdb.connect("C:\\Users\\kalya\\OneDrive\\Documents\\Data Mining Package\\database\\aadhar.duckdb")

for table in ["biometric_data", "demographic_data", "enrolment_data"]:
    print(f"\n{'='*50}")
    print(f"TABLE: {table}")
    print(con.execute(f"DESCRIBE {table}").fetchdf().to_string())
    print(f"\nUnique dates:")
    print(con.execute(f"SELECT DISTINCT date FROM {table} ORDER BY date").fetchdf().to_string())
    print(f"\nRow count: {con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:,}")
    print(f"\nSample:")
    print(con.execute(f"SELECT * FROM {table} LIMIT 3").fetchdf().to_string())