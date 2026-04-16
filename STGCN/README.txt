Synthetic STGCN sample data
===========================

What is included
- raw_csv/biometric_data_preprocessed.csv
- raw_csv/enrolment_data_preprocessed.csv
- shapefile_like/districts.geojson
- district_order.csv
- W_combined.csv
- L_normalised_laplacian.csv
- database/load_into_duckdb.sql

Design choices
- 4 districts: Coimbatore, Erode, Salem, Tiruppur
- 20 weekly records per district
- smooth upward trend with very low variance
- ratios stay realistic and stable
- district names exactly match across CSV and GeoJSON

Why this should give stable predictions
- the target columns (bio_total and enrol_total) change gradually week to week
- district differences are small and consistent
- there are no sudden spikes, drops, or missing weeks

How to use with your files
1. Load the two CSVs into DuckDB using database/load_into_duckdb.sql.
2. For build_district_graph(1).py, pass shapefile_like/districts.geojson as --shp.
3. If you do not want to run graph creation, you can directly use district_order.csv and L_normalised_laplacian.csv for train_stgcn(1).py.
4. Then run the weekly tensor builders followed by train_stgcn.

Suggested commands
python build_district_graph(1).py --shp /mnt/data/stgcn_sample/shapefile_like/districts.geojson --db /path/to/aadhar.duckdb --alpha 0.7 --sigma2 0.5 --epsilon 0.05
python build_biometric_weekly_tensor(2).py
python build_enrolment_weekly_tensor(2).py
python train_stgcn(1).py --tensor biometric_weekly_output/feature_tensor_X.npy --laplacian graph_output/L_normalised_laplacian.csv --districts graph_output/district_order.csv --week_index biometric_weekly_output/week_index.csv --features biometric_weekly_output/tensor_feature_columns.csv --target bio_total --output_dir biometric_stgcn_output --epochs 40 --t_in 8 --t_out 1

Recommended settings for smooth output
- alpha=0.7
- sigma2=0.5
- epsilon=0.05
- epochs=30 to 40
- dropout=0.1 to 0.2
