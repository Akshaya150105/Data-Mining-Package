import os
import sys
import pandas as pd
from collections import defaultdict
from fuzzywuzzy import fuzz, process

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_connection import get_connection


# ---------------------------------------------------
# OFFICIAL STATES
# ---------------------------------------------------

OFFICIAL_STATES = [
    'Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chhattisgarh',
    'Goa','Gujarat','Haryana','Himachal Pradesh','Jharkhand','Karnataka',
    'Kerala','Madhya Pradesh','Maharashtra','Manipur','Meghalaya','Mizoram',
    'Nagaland','Odisha','Punjab','Rajasthan','Sikkim','Tamil Nadu',
    'Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal',
    'Andaman and Nicobar Islands','Chandigarh',
    'Dadra and Nagar Haveli and Daman and Diu',
    'Delhi','Jammu and Kashmir','Ladakh','Lakshadweep','Puducherry'
]


STATE_MANUAL_CORRECTIONS = {
    'westbengal':'West Bengal',
    'west bengal':'West Bengal',
    'west bangal':'West Bengal',
    'orissa':'Odisha',
    'pondicherry':'Puducherry',
    'uttaranchal':'Uttarakhand',
    'chhatisgarh':'Chhattisgarh',
    'tamilnadu':'Tamil Nadu',
    'andaman & nicobar islands':'Andaman and Nicobar Islands',
    'jammu & kashmir':'Jammu and Kashmir'
}


DISTRICT_BLACKLIST_PATTERNS = [
    ('north','south'),
    ('east','west'),
    ('urban','rural'),
]


# ---------------------------------------------------
# BASIC CLEANING
# ---------------------------------------------------

def clean_name(name):

    if pd.isna(name):
        return name

    name = str(name).strip()
    name = name.rstrip('*')
    name = " ".join(name.split())

    return name


# ---------------------------------------------------
# STATE STANDARDIZATION
# ---------------------------------------------------

def preprocess_state(name):

    name = clean_name(name)
    name = name.replace("&","and")

    lower = name.lower()

    if lower in STATE_MANUAL_CORRECTIONS:
        return STATE_MANUAL_CORRECTIONS[lower]

    return name


def fuzzy_match_state(name, threshold=85):

    name = preprocess_state(name)

    if name in OFFICIAL_STATES:
        return name

    match,score = process.extractOne(
        name,
        OFFICIAL_STATES,
        scorer=fuzz.token_sort_ratio
    )

    if score >= threshold:
        return match

    return name.title()


# ---------------------------------------------------
# DISTRICT MERGING RULE
# ---------------------------------------------------

def should_not_merge(a,b):

    a = a.lower()
    b = b.lower()

    for x,y in DISTRICT_BLACKLIST_PATTERNS:

        if (x in a and y in b) or (y in a and x in b):
            return True

    return False


def cluster_names(names,threshold=97):

    clusters=[]
    used=set()

    for name in names:

        if name in used:
            continue

        group=[name]
        used.add(name)

        for other in names:

            if other in used:
                continue

            if should_not_merge(name,other):
                continue

            score=fuzz.token_sort_ratio(name,other)

            if score>=threshold:
                group.append(other)
                used.add(other)

        clusters.append(group)

    return clusters


# ---------------------------------------------------
# MAIN STANDARDIZATION FUNCTION
# ---------------------------------------------------

def smart_standardize(df):

    df=df.copy()

    changes=defaultdict(lambda:defaultdict(int))

    print("\nCleaning basic names")

    df["state"]=df["state"].apply(clean_name)
    df["district"]=df["district"].apply(clean_name)

    print("\nStandardizing states")

    before=df["state"].nunique()

    df["state"]=df["state"].apply(fuzzy_match_state)

    after=df["state"].nunique()

    print("States:",before,"->",after)

    print("\nStandardizing districts")

    before=df["district"].nunique()

    for state in df["state"].unique():

        mask=df["state"]==state

        districts=df.loc[mask,"district"].unique()

        clusters=cluster_names(districts)

        for group in clusters:

            if len(group)>1:

                canonical=max(
                    group,
                    key=lambda x:len(df[(df["state"]==state)&(df["district"]==x)])
                )

                for v in group:

                    if v!=canonical:

                        m=(df["state"]==state)&(df["district"]==v)

                        df.loc[m,"district"]=canonical

                        changes["district"][f"{v}->{canonical}"]+=m.sum()

    after=df["district"].nunique()

    print("Districts:",before,"->",after)

    return df,changes


# ---------------------------------------------------
# PROCESS ALL DUCKDB TABLES
# ---------------------------------------------------

def process_tables():

    con=get_connection()

    tables=[
        "biometric_data",
        "demographic_data",
        "enrolment_data"
    ]

    for table in tables:

        print("\n"+"="*60)
        print("Processing:",table)
        print("="*60)

        df=con.execute(f"SELECT * FROM {table}").df()

        df_clean,changes=smart_standardize(df)

        con.execute(
            f"CREATE OR REPLACE TABLE {table}_clean AS SELECT * FROM df_clean"
        )

        print("Saved cleaned table:",table+"_clean")

    con.close()


# ---------------------------------------------------
# RUN
# ---------------------------------------------------

if __name__=="__main__":

    process_tables()
