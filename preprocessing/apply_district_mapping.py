"""
apply_district_mapping.py – Steps 3-6: District cleaning pipeline
==================================================================
Standardises the 'district' column across all three Aadhaar tables.

  STEP 3 – Basic preprocessing : encoding fix → replace parentheses with
                                  spaces → trim → title-case (hyphens preserved)
  STEP 4 – Fuzzy matching      : per-state match against LGD reference
                                  Excel → state-scoped manual overrides
                                  (manual applied AFTER fuzzy so manual wins)
                                  Writes a timestamped analysis report.
  STEP 5 – Junk deletion       : remove PIN codes, addresses, etc.
  STEP 6 – Final audit         : row/state/district counts + cross-state
                                  collision warning

Prerequisites
-------------
Run apply_state_mapping.py first so that state names are canonical
(the per-state LGD lookup relies on clean state names).

Outputs
-------
  logs/district_mapping_YYYYMMDD_HHMMSS.log
  reports/district_cleaning_YYYYMMDD_HHMMSS.txt
"""

import logging
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_connection import get_connection

# ── Logging & output dirs ──────────────────────────────────────────────────────
LOG_DIR    = Path(__file__).parent.parent / "database" / "logs"
REPORT_DIR = Path(__file__).parent.parent / "database" / "reports"
LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RUN_TS      = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH    = LOG_DIR    / f"district_mapping_{RUN_TS}.log"
REPORT_PATH = REPORT_DIR / f"district_cleaning_{RUN_TS}.txt"

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
TABLES              = ["biometric_data", "demographic_data", "enrolment_data"]
DISTRICT_CONFIDENCE = 80

LGD_PATH = os.path.join(
    PROJECT_ROOT, "Data",
    "LGD - Local Government Directory, Government of India.xlsx"
)

# ── DISTRICT_MANUAL: state-scoped overrides ────────────────────────────────────
# Applied AFTER fuzzy matching → manual always wins.
# Structure: { "Canonical State Name": { "raw_district": "standard_district" } }
#
# NOTE ON PARENTHESES: Step 3 now replaces ( and ) with spaces before title-casing,
# so source values like "Kaimur (Bhabua)" become "Kaimur Bhabua" in the DB.
# Both the original and the post-replacement form are included as keys below
# so the overrides work regardless of which version fuzzy matching left behind.
DISTRICT_MANUAL: dict[str, dict[str, str]] = {
    "Andhra Pradesh": {
        "Spsr Nellore":        "Sri Potti Sriramulu Nellore",
        "K.V.Rangareddy":      "K V Rangareddy",
        "K.V. Rangareddy":     "K V Rangareddy",
        "Rangareddi":          "Ranga Reddy",
        "Anantapur":           "Ananthapuramu",
        "Ananthapur":          "Ananthapuramu",
        "Cuddapah":            "Y S R",
        "Y. S. R":             "Y S R",
        "Visakhapatanam":      "Visakhapatnam",
        "Mahabub Nagar":       "Mahabubnagar",
        "Mahbubnagar":         "Mahabubnagar",
        # "Karim Nagar" is old spelling of "Karimnagar"
        "Karim Nagar":         "Karimnagar",
    },
    "Assam": {
        "North Cachar Hills":  "Dima Hasao",   # officially renamed
        "Sibsagar":            "Sivasagar",
        "Tamulpur District":   "Tamulpur",
    },
    "Bihar": {
        # Original form (before paren-replacement)
        "Kaimur (Bhabua)":     "Kaimur",
        # Post paren-replacement form: "Kaimur (Bhabua)" → "Kaimur Bhabua"
        "Kaimur Bhabua":       "Kaimur",
        "Bhabua":              "Kaimur",
        # Original form
        "Aurangabad Bh":       "Aurangabad",
        # Post paren-replacement: "Aurangabad(Bh)" → "Aurangabad Bh"
        # (already covered above)
        "West Champaran":      "Pashchim Champaran",
        "East Champaran":      "Purba Champaran",
        "Purbi Champaran":     "Purba Champaran",
        "Purana Champaran":    "Purba Champaran",
        "Monghyr":             "Munger",
        "Sheikpura":           "Sheikhpura",
        "Samstipur":           "Samastipur",
        "Purnia":              "Purnea",
    },
    "Chandigarh": {
        # Rupnagar is a Punjab district; rows under Chandigarh are mis-entered
        "Rupnagar":            "Chandigarh",
    },
    "Chhattisgarh": {
        "Gaurella Pendra Marwahi":               "Gaurela-Pendra-Marwahi",
        "Gaurela Pendra Marwahi":                "Gaurela-Pendra-Marwahi",
        "Janjgir Champa":                        "Janjgir-Champa",
        "Janjgir - Champa":                      "Janjgir-Champa",
        "Mohalla Manpur Ambagarh Chowki":        "Mohla-Manpur-Ambagarh Chouki",
        "Mohla Manpur Ambagarh Chouki":          "Mohla-Manpur-Ambagarh Chouki",
        "Mohalla-Manpur-Ambagarh Chowki":        "Mohla-Manpur-Ambagarh Chouki",
        "Manendragarhchirmiribharatpur":          "Manendragarh Chirmiri Bharatpur",
        # Post paren-replacement: "Manendragarh-Chirmiri-Bharatpur(M C B)"
        #   → "Manendragarh-Chirmiri-Bharatpur M C B"
        "Manendragarh-Chirmiri-Bharatpur M C B": "Manendragarh Chirmiri Bharatpur",
        "Dakshin Bastar Dantewada":              "Dantewada",
        "Uttar Bastar Kanker":                   "Kanker",
        "Kabeerdham":                            "Kabirdham",
        "Kawardha":                              "Kabirdham",
    },
    "Dadra and Nagar Haveli and Daman and Diu": {
        # Mis-entered district; belongs to Andaman & Nicobar Islands
        "North And Middle Andaman": "Dadra And Nagar Haveli",
    },
    "Delhi": {
        "North East":          "North East Delhi",
    },
    "Gujarat": {
        "Ahmadabad":           "Ahmedabad",
        "Dohad":               "Dahod",
        "Banas Kantha":        "Banaskantha",
        "Sabar Kantha":        "Sabarkantha",
        "Panch Mahals":        "Panchmahal",
        "Panchmahals":         "Panchmahal",
        "Dangs":               "Dang",
        "The Dangs":           "Dang",
        "Surendra Nagar":      "Surendranagar",
    },
    "Haryana": {
        "Gurgaon":             "Gurugram",
        "Mewat":               "Nuh",
        "Akhera":              "Mahendragarh",
    },
    "Himachal Pradesh": {
        "Lahul & Spiti":       "Lahaul And Spiti",
        "Lahul And Spiti":     "Lahaul And Spiti",
        "Lahul Spiti":         "Lahaul And Spiti",
    },
    "Jammu and Kashmir": {
        "Badgam":              "Budgam",
        "Bandipore":           "Bandipur",
        "Rajauri":             "Rajouri",
        "Poonch":              "Punch",
        "Shupiyan":            "Shopian",
        # Original form
        "Leh (Ladakh)":        "Leh",
        # Post paren-replacement: "Leh (Ladakh)" → "Leh Ladakh"
        "Leh Ladakh":          "Leh",
    },
    "Jharkhand": {
        "Pakaur":              "Pakur",
        "Palamau":             "Palamu",
        "Sahebganj":           "Sahibganj",
        "Koderma":             "Kodarma",
        "East Singhbum":       "Purbi Singhbhum",
        "Saraikela Kharsawan": "Seraikela-Kharsawan",
        "West Singhbhum":      "Pashchimi Singhbhum",
    },
    "Karnataka": {
        "Chickmagalur":        "Chikkamagaluru",
        "Chikmagalur":         "Chikkamagaluru",
        "Tumkur":              "Tumakuru",
        "Shimoga":             "Shivamogga",
        "Mysore":              "Mysuru",
        "Gulbarga":            "Kalaburagi",
        # Post paren-replacement: "Bijapur(Kar)" → "Bijapur Kar"
        "Bijapur Kar":         "Vijayapura",
        "Bijapur":             "Vijayapura",
        "Belgaum":             "Belagavi",
        "Bellary":             "Ballari",
        "Davangere":           "Davanagere",
        "Hasan":               "Hassan",
        "Chamrajnagar":        "Chamarajanagar",
        "Chamrajanagar":       "Chamarajanagar",
        "Ramanagar":           "Ramanagara",
        "Chikkaballapur":      "Chikkaballapura",
        "Bangalore":           "Bengaluru Urban",
        "Bangalore Rural":     "Bengaluru Rural",
        "Bengaluru South":     "Bengaluru Urban",
    },
    "Kerala": {
        "Kasargod":            "Kasaragod",
    },
    "Madhya Pradesh": {
        "Hoshangabad":         "Narmadapuram",   # officially renamed
        "Narsimhapur":         "Narsinghpur",
        "East Nimar":          "Khandwa",
        "West Nimar":          "Khargone",
    },
    "Maharashtra": {
        "Aurangabad":          "Chhatrapati Sambhajinagar",   # officially renamed
        "Osmanabad":           "Dharashiv",                   # officially renamed
        "Ahmadnagar":          "Ahmednagar",
        "Ahmed Nagar":         "Ahmednagar",
        "Ahilyanagar":         "Ahmednagar",
        "Bid":                 "Beed",
        "Gondiya":             "Gondia",
        "Raigarh":             "Raigad",
        # Post paren-replacement: "Raigarh(Mh)" → "Raigarh Mh"
        "Raigarh Mh":          "Raigad",
        "Mumbai City":         "Mumbai",
        "Mumbai Suburban":     "Mumbai",
    },
    "Meghalaya": {
        # Kamrup is an Assam district; Meghalaya rows are mis-entered
        "Kamrup":              "Ri Bhoi",
        # Jaintia Hills was split; old records use undivided name
        "Jaintia Hills":       "East Jaintia Hills",
    },
    "Mizoram": {
        "Mammit":              "Mamit",
    },
    "Odisha": {
        "Anugal":              "Anugul",
        "Angul":               "Anugul",
        "Baleshwar":           "Baleswar",
        "Khorda":              "Khordha",
        "Baudh":               "Boudh",
        "Jagatsinghpur":       "Jagatsinghapur",
        "Jajapur":             "Jajpur",
        "Nabarangapur":        "Nabarangpur",
        "Sonapur":             "Subarnapur",
        "Sundergarh":          "Sundargarh",
        # Post paren-replacement: "Bhadrak(R)" → "Bhadrak R"
        "Bhadrak R":           "Bhadrak",
    },
    "Puducherry": {
        "Pondicherry":         "Puducherry",
    },
    "Punjab": {
        "S A S Nagar Mohali":         "Sahibzada Ajit Singh Nagar",
        "Sas Nagar Mohali":           "Sahibzada Ajit Singh Nagar",
        "S.A.S Nagar":                "Sahibzada Ajit Singh Nagar",
        # Post paren-replacement: "S.A.S Nagar(Mohali)" → "S.A.S Nagar Mohali"
        "S.A.S Nagar Mohali":         "Sahibzada Ajit Singh Nagar",
        # Post paren-replacement: "SAS Nagar (Mohali)" → "Sas Nagar Mohali"
        "Sas Nagar Mohali":           "Sahibzada Ajit Singh Nagar",
        "SAS Nagar Mohali":           "Sahibzada Ajit Singh Nagar",
        "Shaheed Bhagat Singh Nagar": "Shahid Bhagat Singh Nagar",
        "Nawanshahr":                 "Shahid Bhagat Singh Nagar",
        "Muktsar":                    "Sri Muktsar Sahib",
        "Firozpur":                   "Ferozepur",
    },
    "Rajasthan": {
        "Dhaulpur":            "Dholpur",
        "Jalor":               "Jalore",
        "Jhunjhunun":          "Jhunjhunu",
        "Chittaurgarh":        "Chittorgarh",
        "Agar-Malwa":          "Agar Malwa",
    },
    "Sikkim": {
        # Short-form names used in older data
        "East":                "East Sikkim",
        "North":               "North Sikkim",
        "South":               "South Sikkim",
        "West":                "West Sikkim",
    },
    "Tamil Nadu": {
        "Kancheepuram":        "Kanchipuram",
        "Thiruvallur":         "Tiruvallur",
        "Thiruvarur":          "Tiruvarur",
        "Kanniyakumari":       "Kanyakumari",
        "Tuticorin":           "Thoothukudi",
        "Thoothukkudi":        "Thoothukudi",
        "Tirupathur":          "Tirupattur",
        "Villupuram":          "Viluppuram",
    },
    "Telangana": {
        "Jangoan":                  "Jangaon",
        "Medchal Malkajgiri":       "Medchal-Malkajgiri",
        "Medchal-malkajgiri":       "Medchal-Malkajgiri",
        "Medchal?malkajgiri":       "Medchal-Malkajgiri",
        "MedchalâMalkajgiri":       "Medchal-Malkajgiri",   # mojibake variant
        "Medchal Â Malkajgiri":     "Medchal-Malkajgiri",   # mojibake + space variant
        # Post paren-replacement: "Warangal (Urban)" → "Warangal Urban"
        # Fuzzy will catch it but explicit entry is a safety net
        "Warangal Urban":           "Warangal Urban",
        "Yadadri.":                 "Yadadri-Bhuvanagiri",
        # Short form without suffix
        "Yadadri":                  "Yadadri-Bhuvanagiri",
    },
    "Uttar Pradesh": {
        "Allahabad":                  "Prayagraj",           # officially renamed
        "Faizabad":                   "Ayodhya",             # officially renamed
        "Raebareli":                  "Rae Bareli",
        "Mahrajganj":                 "Maharajganj",
        "Jyotiba Phule Nagar":        "Amroha",
        "Jyotiba Phule Nagar *":      "Amroha",
        "Bagpat":                     "Baghpat",
        "Bulandshahar":               "Bulandshahr",
        "Shrawasti":                  "Shravasti",
        "Kheri":                      "Lakhimpur Kheri",
        "Kushi Nagar":                "Kushinagar",
        "Siddharthnagar":             "Siddharth Nagar",
        "Sant Ravidas Nagar":         "Bhadohi",
        "Sant Ravidas Nagar Bhadohi": "Bhadohi",
        "Bara Banki":                 "Barabanki",
    },
    "West Bengal": {
        "Coochbehar":                 "Cooch Behar",
        "Koch Bihar":                 "Cooch Behar",
        "Burdwan":                    "Paschim Bardhaman",
        "Bardhaman":                  "Paschim Bardhaman",
        "Barddhaman":                 "Paschim Bardhaman",
        "Darjiling":                  "Darjeeling",
        "Maldah":                     "Malda",
        "Hawrah":                     "Howrah",
        "Haora":                      "Howrah",
        "Hugli":                      "Hooghly",
        "Hooghiy":                    "Hooghly",
        "North Twenty Four Parganas": "North 24 Parganas",
        "South Twenty Four Parganas": "South 24 Parganas",
        "South 24 Pargana":           "South 24 Parganas",
        "Medinipur West":             "Paschim Medinipur",
        "West Midnapore":             "Paschim Medinipur",
        "East Midnapore":             "Purba Medinipur",
        "West Medinipur":             "Paschim Medinipur",
        "East Midnapur":              "Purba Medinipur",
        "Puruliya":                   "Purulia",
        "Uttar Dinajpur":             "North Dinajpur",
        "Dakshin Dinajpur":           "South Dinajpur",
    },
}

# ── Junk district patterns ─────────────────────────────────────────────────────
# All patterns use (?i) inline flag or re.IGNORECASE on the compiled regex.
# These run against title-cased values (after Step 3), so patterns are
# written to match that casing.
JUNK_DISTRICT_PATTERNS = [
    r"^near\b",           # "Near XYZ hospital / road"
    r"^\d{6}$",               # 6-digit PIN codes
    r"^\d+$",                 # purely numeric
    r"idpl colony",
    r"bally jagachha",
    r"naihati anandabazar",
    r"south dumdum",
    r"domjur",
    r"kadiri road",
    r"balianta",
    r"^bhadrak r\b",
    r"^5th cross$",
]
JUNK_RE = re.compile("|".join(JUNK_DISTRICT_PATTERNS),re.IGNORECASE)

EXPLICIT_JUNK = {
    "Near University Thana", "Near Uday Nagar Nit Garden",
    "Near Dhyana Ashram", "Near Meera Hospital",
    "Kadiri Road", "Idpl Colony", "5Th Cross",
    "Bally Jagachha", "Naihati Anandabazar",
    "South Dumdum M", "Domjur", "Balianta", "Bhadrak R",
    "",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _esc(s: str) -> str:
    return str(s).replace("'", "''")


def _fix_encoding(raw: str) -> str:
    """Fix mojibake and normalise Unicode dashes to plain hyphen."""
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
    """
    Lowercase, strip, collapse whitespace, remove punctuation noise.
    Hyphens are preserved so names like Medchal-Malkajgiri survive.
    """
    if not isinstance(raw, str) or raw == "":
        return ""
    s = _fix_encoding(raw).lower().strip()
    s = re.sub(r"[*?()\[\]{}<>@#$%^&+=|~`\u2019\u2018\u201c\u201d\\/]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _bulk_update(con, table: str, col: str, mapping: dict) -> int:
    """
    Bulk-replace values in `col` of `table` using a dict.
    Returns number of distinct old values that were present before update.
    Uses a uniquely-named temp table — safe for concurrent use.
    """
    if not mapping:
        return 0
    tmp  = f"_map_tmp_{table}_{col}"
    rows = [(k, v) for k, v in mapping.items() if v is not None]
    if not rows:
        return 0
    con.execute(f"DROP TABLE IF EXISTS {tmp}")
    con.execute(f"CREATE TEMP TABLE {tmp} (old_val VARCHAR, new_val VARCHAR)")
    con.executemany(f"INSERT INTO {tmp} VALUES (?, ?)", rows)
    # Count how many distinct old values actually exist (for logging)
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


# ── STEP 3: Basic district preprocessing ──────────────────────────────────────
def step3_preprocess_districts(con) -> None:
    log.info("=" * 60)
    log.info("STEP 3 – Basic district preprocessing")
    log.info("=" * 60)

    counts_before = _row_counts(con)

    for tbl in TABLES:
        rows = con.execute(
            f"SELECT DISTINCT district FROM {tbl} WHERE district IS NOT NULL"
        ).fetchall()
        mapping: dict[str, str] = {}
        for (orig,) in rows:
            # 1. Fix encoding (mojibake, unicode dashes)
            fixed = _fix_encoding(orig)

            # 2. Replace parentheses with spaces BEFORE any other cleaning.
            #    This is critical: the old approach stripped only trailing ")"
            #    leaving corrupt values like "Kaimur (Bhabua" (no closing paren).
            #    Replacing both chars means:
            #      "Kaimur (Bhabua)"  → "Kaimur  Bhabua "  → "Kaimur Bhabua"
            #      "Warangal (Urban)" → "Warangal  Urban "  → "Warangal Urban"
            #      "Bijapur(Kar)"     → "Bijapur Kar "      → "Bijapur Kar"
            #      "Raigarh(Mh)"      → "Raigarh Mh "       → "Raigarh Mh"
            #      "Leh (Ladakh)"     → "Leh  Ladakh "      → "Leh Ladakh"
            cleaned = re.sub(r"[()]", " ", fixed)

            # 3. Collapse whitespace, strip ends
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            # 4. Strip remaining leading/trailing noise chars (keep hyphens)
            cleaned = re.sub(r"^[^a-zA-Z0-9-]+|[^a-zA-Z0-9-]+$", "", cleaned)

            # 5. Title-case
            titled = cleaned.title()

            if orig != titled:
                mapping[orig] = titled

        n = _bulk_update(con, tbl, "district", mapping)
        log.info("  %s: %d district name(s) normalised.", tbl, n)

    _assert_not_increased(counts_before, _row_counts(con), "Step 3")


# ── STEP 4: Fuzzy matching + manual overrides ──────────────────────────────────
def step4_fuzzy_districts(con) -> None:
    log.info("=" * 60)
    log.info("STEP 4 – District fuzzy matching (per-state, LGD reference)")
    log.info("=" * 60)

    counts_before = _row_counts(con)

    # Load LGD reference
    lgd = pd.read_excel(
        LGD_PATH, engine="calamine",
        usecols=["State Name", "District Name (In English)"]
    )
    lgd.columns = ["state", "district"]
    lgd.dropna(inplace=True)
    lgd["state_key"]    = lgd["state"].apply(
        lambda x: clean_text(re.sub(r"\(.*?\)", "", str(x)))
    )
    lgd["district_std"] = lgd["district"].str.strip().str.title()
    lgd_ref: dict[str, list[str]] = {
        k: sorted(g["district_std"].unique())
        for k, g in lgd.groupby("state_key")
    }
    total_lgd = sum(len(v) for v in lgd_ref.values())
    log.info("  LGD loaded: %d states, %d district entries.", len(lgd_ref), total_lgd)

    # Collect unique (state, district) pairs across all tables
    pairs: set[tuple[str, str]] = set()
    for tbl in TABLES:
        pairs.update(con.execute(
            f"SELECT DISTINCT state, district FROM {tbl} "
            f"WHERE state IS NOT NULL AND district IS NOT NULL"
        ).fetchall())
    log.info("  %d unique (state, district) pairs to match.", len(pairs))

    # 4a: Fuzzy match per-state against LGD
    analysis:  list[dict]                = []
    fuzzy_map: dict[tuple[str, str], str] = {}

    for orig_state, orig_district in sorted(pairs):
        state_key  = clean_text(orig_state)
        dist_clean = clean_text(orig_district)

        state_dists = lgd_ref.get(state_key)
        if state_dists:
            res    = process.extractOne(dist_clean, state_dists, scorer=fuzz.WRatio)
            source = "per-state"
        else:
            # State not in LGD → cross-state fallback
            all_d  = [d for v in lgd_ref.values() for d in v]
            res    = process.extractOne(dist_clean, all_d, scorer=fuzz.WRatio)
            source = "cross-state"

        std_dist = res[0] if res and res[1] >= DISTRICT_CONFIDENCE else None
        score    = res[1] if res else 0
        status   = "Success" if std_dist else "Unmatched"

        if std_dist:
            fuzzy_map[(orig_state, orig_district)] = std_dist

        analysis.append({
            "state":   orig_state, "orig":    orig_district,
            "cleaned": dist_clean, "matched": std_dist or "unmatched",
            "score":   round(score, 2), "source": source, "status": status,
        })

    # Apply fuzzy results to DB first
    for tbl in TABLES:
        if not fuzzy_map:
            break
        tmp = f"_dmap_tmp_{tbl}"
        con.execute(f"DROP TABLE IF EXISTS {tmp}")
        con.execute(
            f"CREATE TEMP TABLE {tmp} "
            "(orig_state VARCHAR, orig_dist VARCHAR, std_dist VARCHAR)"
        )
        con.executemany(
            f"INSERT INTO {tmp} VALUES (?, ?, ?)",
            [(k[0], k[1], v) for k, v in fuzzy_map.items()]
        )
        con.execute(f"""
            UPDATE {tbl}
            SET district = {tmp}.std_dist
            FROM {tmp}
            WHERE {tbl}.state    = {tmp}.orig_state
              AND {tbl}.district = {tmp}.orig_dist
              AND {tbl}.district IS DISTINCT FROM {tmp}.std_dist
        """)
        con.execute(f"DROP TABLE IF EXISTS {tmp}")
        log.info("  %s: fuzzy district mapping applied.", tbl)

    # 4b: State-scoped manual overrides ON TOP — manual always wins over fuzzy
    for tbl in TABLES:
        for state, overrides in DISTRICT_MANUAL.items():
            if not overrides:
                continue
            tmp = f"_mmap_tmp_{tbl}"
            con.execute(f"DROP TABLE IF EXISTS {tmp}")
            con.execute(f"CREATE TEMP TABLE {tmp} (old_val VARCHAR, new_val VARCHAR)")
            con.executemany(f"INSERT INTO {tmp} VALUES (?, ?)", list(overrides.items()))
            con.execute(f"""
                UPDATE {tbl}
                SET district = {tmp}.new_val
                FROM {tmp}
                WHERE {tbl}.state    = '{_esc(state)}'
                  AND {tbl}.district = {tmp}.old_val
                  AND {tbl}.district IS DISTINCT FROM {tmp}.new_val
            """)
            con.execute(f"DROP TABLE IF EXISTS {tmp}")
        log.info("  %s: manual overrides applied.", tbl)

    # Write timestamped analysis report
    success_r   = [r for r in analysis if r["status"] == "Success"]
    unmatched_r = [r for r in analysis if r["status"] == "Unmatched"]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("District Cleaning & Fuzzy Matching Report\n")
        f.write(f"Generated: {RUN_TS}\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"LGD reference entries : {total_lgd}\n")
        f.write(f"DB pairs checked      : {len(pairs)}\n")
        f.write(f"Successfully matched  : {len(success_r)}\n")
        f.write(f"Unmatched             : {len(unmatched_r)}\n\n")

        f.write("─" * 120 + "\n")
        f.write(f"{'State':<28} | {'Original':<32} | {'Matched':<32} | {'Score':>6} | Source\n")
        f.write("─" * 120 + "\n")
        for r in analysis:
            f.write(
                f"{r['state'][:26]:<28} | {r['orig'][:30]:<32} | "
                f"{r['matched'][:30]:<32} | {r['score']:>6} | {r['source']}\n"
            )

        f.write("\n\nMerging Strategy\n" + "=" * 60 + "\n")
        groups: dict[tuple, list] = {}
        for r in success_r:
            groups.setdefault((r["state"], r["matched"]), []).append(r["orig"])
        for (st, std), origs in sorted(groups.items()):
            multi = [o for o in origs if o != std]
            if multi:
                f.write(f"[{st}] {std}\n")
                for o in multi:
                    f.write(f"  ← {o}\n")

        f.write("\n\nUnmatched (manual review needed)\n" + "=" * 60 + "\n")
        for r in unmatched_r:
            f.write(f"  [{r['state']}] \"{r['orig']}\" (score {r['score']})\n")

    log.info("  Report written → %s", REPORT_PATH)
    _assert_not_increased(counts_before, _row_counts(con), "Step 4")


# ── STEP 5: Delete junk / invalid rows ────────────────────────────────────────
def step5_delete_junk(con) -> None:
    log.info("=" * 60)
    log.info("STEP 5 – Delete junk / invalid rows")
    log.info("=" * 60)

    counts_before = _row_counts(con)

    for tbl in TABLES:
        deleted = 0
        rows = con.execute(f"SELECT DISTINCT district FROM {tbl}").fetchall()
        junk = {
            r[0] for r in rows
            if r[0] and (JUNK_RE.search(r[0]) or r[0] in EXPLICIT_JUNK)
        }
        for d in junk:
            n = con.execute(
                f"SELECT COUNT(*) FROM {tbl} WHERE district = '{_esc(d)}'"
            ).fetchone()[0]
            if n:
                con.execute(f"DELETE FROM {tbl} WHERE district = '{_esc(d)}'")
                deleted += n
        con.execute(
            f"DELETE FROM {tbl} WHERE district IS NULL OR TRIM(district) = ''"
        )
        log.info("  %s: deleted %d junk/invalid rows.", tbl, deleted)

    _assert_not_increased(counts_before, _row_counts(con), "Step 5")


# ── STEP 6: Final audit ────────────────────────────────────────────────────────
def step6_audit(con) -> None:
    log.info("=" * 60)
    log.info("STEP 6 – Final audit")
    log.info("=" * 60)

    for tbl in TABLES:
        n   = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        u_s = con.execute(f"SELECT COUNT(DISTINCT state)    FROM {tbl}").fetchone()[0]
        u_d = con.execute(f"SELECT COUNT(DISTINCT district) FROM {tbl}").fetchone()[0]
        log.info("  %-22s : %10d rows | %3d states | %4d districts", tbl, n, u_s, u_d)

    cross = con.execute("""
        SELECT district, COUNT(DISTINCT state) AS n_states
        FROM (
            SELECT DISTINCT state, district FROM biometric_data  UNION
            SELECT DISTINCT state, district FROM demographic_data UNION
            SELECT DISTINCT state, district FROM enrolment_data
        ) sub
        GROUP BY district
        HAVING COUNT(DISTINCT state) > 1
        ORDER BY n_states DESC, district
    """).fetchall()

    if cross:
        log.warning("  %d district name(s) appear in multiple states:", len(cross))
        for dist, n_states in cross[:20]:
            states = con.execute(f"""
                SELECT DISTINCT state FROM (
                    SELECT state, district FROM biometric_data   UNION ALL
                    SELECT state, district FROM demographic_data  UNION ALL
                    SELECT state, district FROM enrolment_data
                ) s WHERE district = '{_esc(dist)}'
            """).fetchall()
            log.warning("    %-30s  (%d states: %s)",
                        dist, n_states, ", ".join(r[0] for r in states))
    else:
        log.info("  No cross-state district collisions detected.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    con = get_connection()

    log.info("District mapping pipeline start – %s", RUN_TS)
    log.info("Log    → %s", LOG_PATH)
    log.info("Report → %s", REPORT_PATH)

    step3_preprocess_districts(con)
    step4_fuzzy_districts(con)
    step5_delete_junk(con)
    step6_audit(con)

    log.info("=" * 60)
    log.info("District mapping pipeline complete!")
    log.info("=" * 60)

    con.close()