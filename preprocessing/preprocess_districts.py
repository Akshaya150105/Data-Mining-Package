import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_connection import get_connection

def preprocess_district_name(district):
    if pd.isna(district):
        return ""
    # Convert to lowercase and strip leading/trailing spaces
    cleaned = str(district).lower().strip()
    # Normalize multiple spaces into a single space
    cleaned = ' '.join(cleaned.split())
    # Optionally remove asterisks or common trailing artifacts seen in output
    for char in ['*', '?', '(', ')', '.', ',', '-', '_']:
        cleaned = cleaned.replace(char, ' ')
    cleaned = ' '.join(cleaned.split())
    
    # Capitalize the first letter of each word to look nicer
    return cleaned.title()

def apply_basic_district_preprocessing():
    con = get_connection()
    tables = ['biometric_data', 'demographic_data', 'enrolment_data']
    
    print("Starting basic preprocessing of district names (lowercase, trim, normalize spaces)...")
    
    for table in tables:
        print(f"\nProcessing {table}...")
        
        # Check if table has rows
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count == 0:
            print("  Table is empty, skipping.")
            continue
            
        # Get unique original districts
        try:
            results = con.execute(f"SELECT DISTINCT district FROM {table} WHERE district IS NOT NULL").fetchall()
            original_districts = [r[0] for r in results]
            
            # Create mapping
            mapping = {}
            for orig in original_districts:
                cleaned = preprocess_district_name(orig)
                # Only add to mapping if it actually changed
                if orig != cleaned:
                    mapping[orig] = cleaned
                    
            if not mapping:
                print("  No districts needed basic preprocessing formatting.")
                continue
                
            print(f"  Applying basic formatting to {len(mapping)} distinct district name variants...")
            
            # Apply mapping using temp table approach (much faster for large datasets)
            con.execute("DROP TABLE IF EXISTS dist_mapping_temp")
            con.execute("CREATE TEMP TABLE dist_mapping_temp (old_dist VARCHAR, new_dist VARCHAR)")
            
            for old_d, new_d in mapping.items():
                old_escaped = old_d.replace("'", "''")
                new_escaped = new_d.replace("'", "''")
                con.execute(f"INSERT INTO dist_mapping_temp VALUES ('{old_escaped}', '{new_escaped}')")
                
            query = f"""
                UPDATE {table}
                SET district = dist_mapping_temp.new_dist
                FROM dist_mapping_temp
                WHERE {table}.district = dist_mapping_temp.old_dist
            """
            con.execute(query)
            con.execute("DROP TABLE IF EXISTS dist_mapping_temp")
            
            print(f"  Successfully standardized spacing and casing in {table}.")
            
        except Exception as e:
            print(f"  Error processing {table}: {e}")

    print("\nBasic district preprocessing complete!")
    con.close()

if __name__ == "__main__":
    apply_basic_district_preprocessing()
