import duckdb

con = duckdb.connect("aadhar.duckdb")  # persistent DB file
# Enrolment data
con.execute("""
CREATE OR REPLACE TABLE enrolment AS
SELECT *
FROM read_csv_auto('../Data/api_data_aadhar_enrolment/*.csv');
""")

# Demographic data
con.execute("""
CREATE OR REPLACE TABLE demographic AS
SELECT *
FROM read_csv_auto('../Data/api_data_aadhar_demographic/*.csv');
""")

# Biometric data
con.execute("""
CREATE OR REPLACE TABLE biometric AS
SELECT *
FROM read_csv_auto('../Data/api_data_aadhar_biometric/*.csv');
""")
print(con.execute("SHOW TABLES").fetchdf())
con.execute("DESCRIBE enrolment").df()
con.execute("DESCRIBE demographic").df()
con.execute("DESCRIBE biometric").df()
