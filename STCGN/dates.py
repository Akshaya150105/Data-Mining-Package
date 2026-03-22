import duckdb

con = duckdb.connect("C:\\Users\\kalya\\OneDrive\\Documents\\Data Mining Package\\database\\aadhar.duckdb")

for table in ["biometric_data_preprocessed", "demographic_data_preprocessed", "enrolment_data_preprocessed"]:
    print(f"\n{'='*50}")
    print(f"TABLE: {table}")
    print(f"\nUnique dates:")
    print(con.execute(f"SELECT DISTINCT date FROM {table} ORDER BY date").fetchdf().to_string())
    print(f"\nRow count: {con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:,}")


# After running the above, find common dates manually or automatically:
bio_dates   = set(con.execute("SELECT DISTINCT CAST(date AS VARCHAR) AS date FROM biometric_data_preprocessed").df()['date'])
demo_dates  = set(con.execute("SELECT DISTINCT CAST(date AS VARCHAR) AS date FROM demographic_data_preprocessed").df()['date'])
enrol_dates = set(con.execute("SELECT DISTINCT CAST(date AS VARCHAR) AS date FROM enrolment_data_preprocessed").df()['date'])

# Dates present in ALL 3 tables
MONTHLY_DATES = sorted(bio_dates & demo_dates & enrol_dates)
print("Common dates across all 3 preprocessed tables:")
for d in MONTHLY_DATES:
    print(f"  {d}")

con.close()
