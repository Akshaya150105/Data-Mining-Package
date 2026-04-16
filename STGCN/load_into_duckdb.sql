-- Run this inside DuckDB
CREATE OR REPLACE TABLE biometric_data_preprocessed_stgcn (
    id VARCHAR,
    date DATE,
    state VARCHAR,
    district VARCHAR,
    pincode VARCHAR,
    bio_age_5_17 DOUBLE,
    bio_age_17_ DOUBLE,
    bio_total DOUBLE,
    age_5_ratio DOUBLE,
    age_17_ratio DOUBLE,
    dependency_ratio DOUBLE
);

CREATE OR REPLACE TABLE enrolment_data_preprocessed_stgcn (
    id VARCHAR,
    date DATE,
    state VARCHAR,
    district VARCHAR,
    pincode VARCHAR,
    age_0_5 DOUBLE,
    age_5_17 DOUBLE,
    age_18_greater DOUBLE,
    enrol_total DOUBLE,
    enrol_minor_ratio DOUBLE,
    enrol_adult_ratio DOUBLE
);

COPY biometric_data_preprocessed FROM 'biometric_data_preprocessed.csv' (AUTO_DETECT TRUE, HEADER TRUE);
COPY enrolment_data_preprocessed FROM 'enrolment_data_preprocessed.csv' (AUTO_DETECT TRUE, HEADER TRUE);
