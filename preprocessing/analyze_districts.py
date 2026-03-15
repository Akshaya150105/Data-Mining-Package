import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_connection import get_connection

def analyze_districts():
    con = get_connection()
    tables = ['biometric_data', 'demographic_data', 'enrolment_data']
    output_filename = os.path.join(os.path.dirname(__file__), 'district_analysis.txt')

    print(f"Starting district analysis across {len(tables)} tables...")

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('District Analysis from Database Tables\n')
        f.write('======================================\n\n')
        
        all_districts = set()
        
        for table in tables:
            # Check if table has data
            count_query = f"SELECT COUNT(*) FROM {table}"
            count = con.execute(count_query).fetchone()[0]
            
            f.write(f'Table: {table} (Total Rows: {count:,})\n')
            f.write('-' * (len(table) + 25) + '\n')
            
            if count == 0:
                f.write('  No data in table.\n\n')
                print(f"Table {table} is empty.")
                continue
                
            # Get unique districts and their counts
            query = f"""
                SELECT state, district, COUNT(*) as record_count 
                FROM {table} 
                GROUP BY state, district 
                ORDER BY state, district
            """
            df = con.execute(query).df()
            
            f.write(f'  Unique Districts in {table}: {len(df)}\n')
            print(f"Table {table} has {len(df)} unique districts.")
            
            # Add to overall set
            for _, row in df.iterrows():
                # Store as a tuple of (state, district) for cleaner processing later
                all_districts.add((row['state'], row['district']))
                
            f.write('\n')
            
        f.write('Overall Unique Districts across all tables\n')
        f.write('==========================================\n')
        f.write(f'Total Unique State-District Combinations: {len(all_districts)}\n\n')
        
        # Group the overall unique districts by state for a clean list
        overall_by_state = {}
        for state, district in all_districts:
            if state not in overall_by_state:
                overall_by_state[state] = []
            overall_by_state[state].append(district)
            
        for state in sorted(overall_by_state.keys()):
            districts = sorted(overall_by_state[state])
            f.write(f'{state} ({len(districts)} districts):\n')
            for dist in districts:
                f.write(f'  - {dist}\n')
            f.write('\n')

    con.close()
    print(f"\nAnalysis complete! Results saved to:\n{output_filename}")

if __name__ == "__main__":
    analyze_districts()
