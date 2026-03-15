"""
apply_state_mapping.py – Step 2: State name standardisation
============================================================
Cleans and standardises the 'state' column across all three
Aadhaar tables using:
  1. Encoding repair  (mojibake / Latin-1 mis-read)
  2. Fuzzy matching   (RapidFuzz WRatio, threshold 80)
  3. Typo overrides   (old names → canonical names)
  4. Junk deletion    (city/locality values in state column)
  5. Row-count audit  (assert no rows were added)

Usage
-----
    cd database
    python apply_state_mapping.py
"""

import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from datetime import datetime

from rapidfuzz import fuzz, process

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_connection import get_connection

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "database" / "logs"
LOG_DIR.mkdir(exist_ok=True)
RUN_TS   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"state_mapping_{RUN_TS}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TABLES            = ["biometric_data", "demographic_data", "enrolment_data"]
STATE_CONFIDENCE  = 80

# Canonical list of 36 Indian states / UTs  (lowercase for fuzzy matching)
STANDARD_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "andaman and nicobar islands", "chandigarh",
    "dadra and nagar haveli and daman and diu",
    "delhi", "jammu and kashmir", "ladakh", "lakshadweep", "puducherry",
]

# Genuine old / mis-spelled state names → correct canonical name
STATE_TYPOS = {
    "Orissa":      "Odisha",
    "Pondicherry": "Puducherry",
    "Uttaranchal": "Uttarakhand",
}

# City / locality values found in the state column → delete the row
STATE_CITIES_TO_DELETE = {
    "100000",               # numeric junk
    "BALANAGAR",            # locality in Telangana
    "Darbhanga",            # Bihar district leaked into state
    "Jaipur",               # Rajasthan city
    "Madanapalle",          # AP city
    "Nagpur",               # MH city
    "Puttenahalli",         # KA locality
    "Raja Annamalai Puram", # TN locality
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _esc(s: str) -> str:
    return str(s).replace("'", "''")


def _fix_encoding(raw: str) -> str:
    """Repair mojibake and normalise to Unicode NFC."""
    if not isinstance(raw, str):
        return raw
    try:
        fixed = raw.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        fixed = raw
    fixed = unicodedata.normalize("NFC", fixed)
    fixed = re.sub(r"[\u2013\u2014\u2212]", "-", fixed)
    return fixed


def clean_text(raw: str) -> str:
    """Lowercase + strip + collapse whitespace + remove noise chars."""
    if not isinstance(raw, str) or raw == "":
        return ""
    s = _fix_encoding(raw).lower().strip()
    s = re.sub(r"[*?()\[\]{}<>@#$%^&+=|~`\u2019\u2018\u201c\u201d\\/]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _bulk_update(con, table: str, col: str, mapping: dict) -> int:
    """Batch-update *col* of *table* using a mapping dict. Returns # distinct
    old values that were matched (proxy for rows touched)."""
    if not mapping:
        return 0
    tmp  = f"_map_tmp_{table}_{col}"
    rows = [(k, v) for k, v in mapping.items() if v is not None]
    if not rows:
        return 0
    con.execute(f"DROP TABLE IF EXISTS {tmp}")
    con.execute(f"CREATE TEMP TABLE {tmp} (old_val VARCHAR, new_val VARCHAR)")
    con.executemany(f"INSERT INTO {tmp} VALUES (?, ?)", rows)
    matched = con.execute(f"""
        SELECT COUNT(DISTINCT {tmp}.old_val)
        FROM   {tmp}
        JOIN   {table} ON {table}.{col} = {tmp}.old_val
    """).fetchone()[0]
    con.execute(f"""
        UPDATE {table}
        SET    {col} = {tmp}.new_val
        FROM   {tmp}
        WHERE  {table}.{col} = {tmp}.old_val
          AND  {table}.{col} IS DISTINCT FROM {tmp}.new_val
    """)
    con.execute(f"DROP TABLE IF EXISTS {tmp}")
    return matched


def _row_counts(con) -> dict[str, int]:
    return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


def _assert_not_increased(before: dict, after: dict, step: str) -> None:
    for t in TABLES:
        if after[t] > before[t]:
            raise RuntimeError(
                f"Row-count validation FAILED at {step}: "
                f"{t} grew from {before[t]:,} → {after[t]:,}"
            )
        diff = before[t] - after[t]
        if diff:
            log.info("    %s: -%d rows removed in %s", t, diff, step)


# ── Main pipeline step ─────────────────────────────────────────────────────────
def standardise_states(con) -> None:
    log.info("=" * 60)
    log.info("STEP 2 – Standardise state names")
    log.info("=" * 60)

    counts_before = _row_counts(con)

    # 1) Collect raw state values
    raw_states: set[str] = set()
    for tbl in TABLES:
        for (s,) in con.execute(
                f"SELECT DISTINCT state FROM {tbl} WHERE state IS NOT NULL").fetchall():
            raw_states.add(s)

    # 2) Fix encoding first, then refresh
    encoding_map = {s: _fix_encoding(s) for s in raw_states if _fix_encoding(s) != s}
    for tbl in TABLES:
        _bulk_update(con, tbl, "state", encoding_map)

    raw_states = set()
    for tbl in TABLES:
        for (s,) in con.execute(
                f"SELECT DISTINCT state FROM {tbl} WHERE state IS NOT NULL").fetchall():
            raw_states.add(s)

    # 3) Fuzzy-match → canonical
    fuzzy_map:  dict[str, str] = {}
    unresolved: list[str]      = []
    for orig in raw_states:
        cleaned = clean_text(orig)
        if not cleaned:
            continue
        res = process.extractOne(cleaned, STANDARD_STATES, scorer=fuzz.WRatio)
        if res and res[1] >= STATE_CONFIDENCE:
            fuzzy_map[orig] = res[0].title().replace(" And ", " and ")
        else:
            unresolved.append(orig)

    # 4) Apply typo overrides on top
    for orig, std in STATE_TYPOS.items():
        fuzzy_map[orig] = std

    for tbl in TABLES:
        n = _bulk_update(con, tbl, "state", fuzzy_map)
        log.info("  %s: %d distinct state value(s) remapped.", tbl, n)

    # 5) Delete city-as-state rows
    for tbl in TABLES:
        for bad in STATE_CITIES_TO_DELETE:
            n = con.execute(
                f"SELECT COUNT(*) FROM {tbl} WHERE state = '{_esc(bad)}'"
            ).fetchone()[0]
            if n:
                con.execute(f"DELETE FROM {tbl} WHERE state = '{_esc(bad)}'")
                log.info("  Deleted %d rows with state='%s' from %s.", n, bad, tbl)

        # Purge anything still not in canonical list
        canonical = [s.title().replace(" And ", " and ") for s in STANDARD_STATES]
        canon_sql = ", ".join(f"'{_esc(s)}'" for s in canonical)
        con.execute(f"DELETE FROM {tbl} WHERE state NOT IN ({canon_sql})")

    if unresolved:
        log.warning("  %d state(s) could not be resolved: %s", len(unresolved), unresolved)

    _assert_not_increased(counts_before, _row_counts(con), "Step 2")
    log.info("  State standardisation complete.")


if __name__ == "__main__":
    con = get_connection()
    standardise_states(con)
    log.info("Log → %s", LOG_PATH)
    con.close()
