import duckdb
import glob

con = duckdb.connect('aadhar.duckdb')

con.execute("""
CREATE OR REPLACE TABLE biometric_data (
    id           UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
    date         DATE         NOT NULL,
    state        VARCHAR      NOT NULL,
    district     VARCHAR      NOT NULL,
    pincode      VARCHAR(6)  NOT NULL,          -- ← changed to string
    bio_age_5_17 INTEGER       NOT NULL CHECK (bio_age_5_17 >= 0 ),
    bio_age_17_  INTEGER       NOT NULL CHECK (bio_age_17_  >= 0)
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
    demo_age_17_  INTEGER      NOT NULL CHECK (demo_age_17_  >= 0)
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

print("Biometric")
csv_files = glob.glob('../Data/api_data_aadhar_biometric/*.csv')
if not csv_files:
    print("No CSV files found in the directory!")
    print("Checked path:", '../Data/api_data_aadhar_biometric/')
    con.close()
    exit()

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print("  ", f)

total_rows = 0

for csv_path in csv_files:
    print(f"\nLoading: {csv_path}")
    try:
        con.execute(f"""
            COPY biometric_data (
                date,
                state,
                district,
                pincode,
                bio_age_5_17,
                bio_age_17_
            )
            FROM '{csv_path}'
            (
                DELIMITER ',',
                HEADER TRUE,
                quote '"',
                escape '"',
                nullstr ''
            );
        """)
        
        new_rows = con.execute("SELECT COUNT(*) FROM biometric_data").fetchone()[0]

        total_rows += new_rows
        print(f"  → inserted {new_rows:,} rows")
        
    except Exception as e:
        print(f"  ERROR loading {csv_path}:")
        print(" ", str(e))

print(f"\nTotal rows loaded: {total_rows:,}")

print("Demographic")
csv_files = glob.glob('../Data/api_data_aadhar_demographic/*.csv')
if not csv_files:
    print("No CSV files found in the directory!")
    print("Checked path:", '../Data/api_data_aadhar_demographic/')
    con.close()
    exit()

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print("  ", f)

total_rows = 0

for csv_path in csv_files:
    print(f"\nLoading: {csv_path}")
    try:
        con.execute(f"""
            COPY demographic_data (
                date,
                state,
                district,
                pincode,
                demo_age_5_17,
                demo_age_17_
            )
            FROM '{csv_path}'
            (DELIMITER ',', HEADER TRUE, quote '"', escape '"', nullstr '');
        """)
        
        new_rows = con.execute("SELECT COUNT(*) FROM demographic_data").fetchone()[0]

        total_rows += new_rows
        print(f"  → inserted {new_rows:,} rows")
        
    except Exception as e:
        print(f"  ERROR loading {csv_path}:")
        print(" ", str(e))

print(f"\nTotal rows loaded: {total_rows:,}")

print("Enrolment")

csv_files = glob.glob('../Data/api_data_aadhar_enrolment/*.csv')
if not csv_files:
    print("No CSV files found in the directory!")
    print("Checked path:", '../Data/api_data_aadhar_enrolment/')
    con.close()
    exit()

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print("  ", f)

total_rows = 0

for csv_path in csv_files:
    print(f"\nLoading: {csv_path}")
    try:
        con.execute(f"""
            COPY enrolment_data (
                date,
                state,
                district,
                pincode,
                age_0_5,
                age_5_17,
                age_18_greater
            )
            FROM '{csv_path}'
            (DELIMITER ',', HEADER TRUE, quote '"', escape '"', nullstr '');
        """)
        
        new_rows = con.execute("SELECT COUNT(*) FROM enrolment_data").fetchone()[0]

        total_rows += new_rows
        print(f"  → inserted {new_rows:,} rows")
        
    except Exception as e:
        print(f"  ERROR loading {csv_path}:")
        print(" ", str(e))

print(f"\nTotal rows loaded: {total_rows:,}")

# Quick checks
print("Biometric\nSample rows:")
con.sql("SELECT * FROM biometric_data LIMIT 5").show()
print("\nRow count:", con.sql("SELECT COUNT(*) FROM biometric_data").fetchone()[0])

print("Demographic\nSample rows:")
con.sql("SELECT * FROM demographic_data LIMIT 5").show()
print("\nRow count:", con.sql("SELECT COUNT(*) FROM demographic_data").fetchone()[0])

print("Enrolment\nSample rows:")
con.sql("SELECT * FROM enrolment_data LIMIT 5").show()

print("\nRow count:", con.sql("SELECT COUNT(*) FROM enrolment_data").fetchone()[0])
con.close()
