import duckdb
con = duckdb.connect()

df=con.execute("""
DESCRIBE SELECT *
FROM read_csv_auto('../Data/api_data_aadhar_enrolment/*.csv')
""").df()
print(df)