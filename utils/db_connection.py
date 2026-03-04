import duckdb
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(PROJECT_ROOT, "database", "aadhar.duckdb")

def get_connection():
    return duckdb.connect(DB_PATH)