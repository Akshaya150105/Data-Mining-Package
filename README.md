# Aadhaar Enrolment Data Mining & Forecasting

> A complete data mining pipeline on UIDAI's open Aadhaar dataset — covering 775 districts, ~5M records and 70 time steps across 9 months (April – December 2025).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Stage 1 — Data Ingestion & Storage](#stage-1--data-ingestion--storage)
- [Stage 2 — Preprocessing & Feature Engineering](#stage-2--preprocessing--feature-engineering)
- [Stage 3 — District Clustering](#stage-3--district-clustering)
- [Stage 4 — Graph Construction](#stage-4--graph-construction)
- [Stage 5 — STGCN Forecasting](#stage-5--stgcn-forecasting)
- [Stage 6 — Analysis & Visualisation](#stage-6--analysis--visualisation)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Results Summary](#results-summary)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## Project Overview

Aadhaar is the world's largest biometric identity programme. The raw enrolment data is an administrative log file — it records *what* happened but gives no insight into *where* adoption is failing, *when* patterns shift, or *why* certain districts consistently lag behind their neighbours.

This project builds a complete data mining pipeline to extract geographic, demographic and temporal knowledge from that data:

- **Who** is lagging — district clustering identifies 5 behavioural archetypes
- **Where** patterns cluster — spatial autocorrelation via Moran's I
- **When** behaviour shifts — time series analysis across 70 dates
- **What next** — STGCN graph neural network forecasts future enrolment per district

---

## Dataset

| Table | Records | Dates | Key columns |
|---|---|---|---|
| `biometric_data` | 1,861,093 | 89 | `bio_age_5_17`, `bio_age_17_`, `state`, `district`, `pincode` |
| `demographic_data` | 2,071,675 | 95 | `demo_age_5_17`, `demo_age_17_`, `state`, `district` |
| `enrolment_data` | 1,006,007 | 92 | `age_0_5`, `age_5_17`, `age_18_greater`, `state`, `district` |

**Common dates across all 3 tables: 70**
- Monthly snapshots: Apr–Jul 2025
- Daily records: Sep 1–20, Oct 13–31, Nov 1–25, Dec 15–29 2025

**Source:** UIDAI Aadhaar open data portal

---

## Project Structure

```
Data Mining Package/
│
├── database/
│   └── aadhar.duckdb                  # All raw + preprocessed tables
│
├── Adjacency_marix/
│   └── 2011_Dist.shp                  # India district shapefile
│
├── adjacency_output/                  # Graph construction outputs
│   ├── W_combined.csv
│   ├── L_normalised_laplacian.csv
│   ├── feature_tensor_X.npy
│   └── district_order.csv
│
├── STCGN/                             # STGCN model directory
│   ├── stgcn_train.py
│   ├── stgcn_output/
│   │   ├── best_model.pt
│   │   ├── metrics.txt
│   │   └── per_district_error.csv
│   └── adjacency_output/              # Copy of graph outputs
│
├── Insights/                          # Streamlit dashboard + analysis scripts
│   ├── app.py                         # Main dashboard
│   ├── preprocessing.py
│   ├── weighted_adjacency_matrix.py
│   ├── clustering.py
│   ├── state_comparison.py
│   ├── time_series.py
│   ├── table_analysis.py
│   ├── spatial_autocorr.py
│   ├── stgcn_forecast.py
│   │
│   ├── clustering_output/
│   ├── state_output/
│   ├── timeseries_output/
│   ├── table_output/
│   ├── spatial_output/
│   └── forecast_output/
```

---

## Pipeline Overview

```
Raw CSVs → DuckDB → Preprocessing → Feature Engineering
                                           ↓
                              Clustering (K-Means + DBSCAN)
                                           ↓
                              Graph Construction (W_combined + Laplacian)
                                           ↓
                              STGCN Training [T=70, N=945, C=7]
                                           ↓
                              Forecasting + Streamlit Dashboard
```

---

## Stage 1 — Data Ingestion & Storage

**Script:** `preprocessing.py`

Raw Aadhaar CSVs are loaded into **DuckDB** — chosen because it is embedded (no server), columnar (fast analytical queries), and integrates directly with pandas via `.fetchdf()`.

All 3 raw tables are stored in `aadhar.duckdb`. Preprocessed outputs are written back to the same file as new tables.

---

## Stage 2 — Preprocessing & Feature Engineering

**Script:** `preprocessing.py`  
**Output tables:** `biometric_data_preprocessed`, `demographic_data_preprocessed`, `enrolment_data_preprocessed`

### Preprocessing steps

| Step | What it does |
|---|---|
| `validate_data()` | Schema checks, type enforcement |
| `convert_date_column()` | Parses TIMESTAMP → pure DATE, fixes timezone issues |
| `handle_duplicates()` | Removes duplicate rows on (district, date, pincode) |
| `handle_missing_values()` | Forward-fill within district, zero-fill where no prior value |

### Engineered features (18 per district)

| Category | Features |
|---|---|
| **Ratio features** | `age_5_ratio`, `age_17_ratio`, `dependency_ratio`, `enrol_minor_ratio`, `enrol_adult_ratio` |
| **Aggregated** | `state_total`, `district_total`, `district_rank_in_state` |
| **Time features** | `year`, `month`, `quarter`, `day_of_week`, `is_weekend` |
| **Growth features** | `daily_change`, `daily_pct_change`, `{col}_7day_avg`, `{col}_7day_std` |

---

## Stage 3 — District Clustering

**Script:** `clustering.py`  
**Output:** `clustering_output/`

### Method

- **K-Means** on 18 engineered ratio + growth features (raw counts excluded to prevent size bias)
- **Optimal K** selected via elbow curve + silhouette score → **K=5**
- **DBSCAN** run in parallel to detect genuine outliers without forced assignment
- Results visualised on India's 2011 district shapefile with fuzzy name matching (RapidFuzz, threshold=82)

### Cluster profiles

| Cluster | Label | Description | Region |
|---|---|---|---|
| 0 | Mainstream | Balanced ratios, stable trends | South India, NE states |
| 1 | High dependency | High child-to-adult ratio | UP, Bihar, MP (BIMARU belt) |
| 2 | High growth | Campaign-driven spikes, high volatility | Scattered |
| 3 | Adult dominant | High adult ratio, near-saturated | Gujarat, South India |
| 4 | Restricted/border | Low ratios, high volatility, different administration | J&K, Ladakh |

**DBSCAN:** 726 mainstream · 9 metro outlier cluster · 40 noise districts

---

## Stage 4 — Graph Construction

**Script:** `weighted_adjacency_matrix.py`  
**Output:** `adjacency_output/`

### Adjacency matrix

```
W_combined = 0.5 × W_distance + 0.5 × W_similarity
```

| Matrix | Method |
|---|---|
| `W_distance` | Gaussian kernel on centroid distances (UTM zone 43N, σ²=0.1, ε=0.1) |
| `W_similarity` | Cosine similarity of district feature vectors (monthly dates) |
| `W_combined` | Weighted combination (α=0.5) |
| `L_normalised` | Normalised Laplacian: L = I − D⁻¹/² W D⁻¹/² |

### Feature tensor

```
X shape: [T=70, N=945, C=7]
```

| C index | Feature |
|---|---|
| 0 | `bio_age_5_17` |
| 1 | `bio_age_17_` |
| 2 | `bio_total` |
| 3 | `demo_age_5_17` |
| 4 | `demo_age_17_` |
| 5 | `enrol_minor` (age_5_17) |
| 6 | `enrol_adult` (age_18_greater) |

---

## Stage 5 — STGCN Forecasting

**Scripts:** `stgcn_train.py`, `stgcn_forecast.py`  
**Output:** `stgcn_output/`, `forecast_output/`

### Architecture

```
Input X [B, T_in=6, N=945, C=7]
    ↓
ST-Conv Block 1  (Temporal → ChebConv K=4 → Temporal, Kt=2)
    ↓
ST-Conv Block 2  (Temporal → ChebConv K=4 → Temporal, Kt=2)
    ↓
Output FC layer
    ↓
Prediction [B, 1, N=945, C=7]
```

### Training configuration

| Parameter | Value |
|---|---|
| Input window (T_in) | 6 time steps |
| Chebyshev order (K) | 4 |
| Temporal kernel (Kt) | 2 |
| Channels (c_out) | 64 |
| Normalisation | Z-score per feature |
| Loss | MSE |
| Optimiser | Adam (lr=0.001) |
| Scheduler | CosineAnnealingWarmRestarts |
| Max epochs | 200 |
| Early stopping | patience=25 |
| Training sequences | 37 |

### Evaluation results

| Feature | MAE | RMSE | R² |
|---|---|---|---|
| enrol_total | 0.0838 | 0.1414 | 0.252 |
| bio_total | 0.0854 | 0.1260 | 0.214 |
| enrol_minor_ratio | 0.1176 | 0.1793 | 0.319 |
| enrol_adult_ratio | 0.1605 | 0.3047 | 0.406 |
| bio_dependency | 0.1180 | 0.2234 | 0.451 |
| **Mean (ex growth)** | **0.1067** | **0.2272** | **0.316** |

> All metrics on z-scored scale. `enrol_growth_pct` excluded from R² mean (R²=0.009 — inherently noisy derivative signal).

### Forecasting

`stgcn_forecast.py` autoregressively predicts N steps beyond the dataset using the saved checkpoint. Each prediction feeds back as input for the next step.

```bash
python stgcn_forecast.py --steps 6
```

---

## Stage 6 — Analysis & Visualisation

### State comparison (`state_comparison.py`)
Rankings by total enrolment · adult vs minor stacked bar · dependency heatmap (state × month) · size vs growth bubble chart

### Time series (`time_series.py`)
National daily trend with 7-day smoothing · top 10 states · state × date heatmap · month-over-month growth · volatility ranking

### Table analysis (`table_analysis.py`)
Complete stats for each table — distributions, top districts, state aggregations, cross-table correlation matrix (16 charts)

### Spatial autocorrelation (`spatial_autocorr.py`)
Global Moran's I with 999-permutation significance test · Local Moran's I (LISA) classifying each district as HH/LL/HL/LH/NS · LISA choropleth map

---

## Streamlit Dashboard

**Run:**
```bash
cd "Data Mining Package/Insights"
streamlit run app.py
```

### Pages

| Page | What it shows |
|---|---|
| Clustering Analysis | K-Means map · DBSCAN map · cluster profiles · elbow/silhouette · PCA scatter |
| Table Analysis | Biometric · Enrolment · Demographic tabs — full stats, distributions, heatmaps |
| State Comparison | Rankings · ratios · growth · dependency heatmap · scatter |
| Time Series Trends | National trend · top 10 states · heatmap · MoM growth · volatility |
| Spatial Autocorrelation | LISA map · Moran scatter · Global I by feature · district table |
| District Deep-Dive | Select any district — KPIs · time series · state peer comparison |
| STGCN Results | R² by feature · loss curve · predicted vs actual · district errors |
| Forecast | National forecast · top districts · all features · change heatmap |

---

## Results Summary

| Analysis | Key finding |
|---|---|
| Clustering | 5 distinct district archetypes — BIMARU high-dependency belt clearly separates from adult-saturated South India |
| Spatial autocorrelation | Global Moran's I > 0 (statistically significant) — enrolment clusters geographically, validating the STGCN graph |
| Time series | Campaign-driven spikes visible in cluster 2 districts — volatility 5× higher than mainstream districts |
| STGCN | R²=0.32 on stable features across 945 districts simultaneously — cluster 2 and cluster 4 districts have highest prediction error, consistent with clustering results |
| Forecast | Autoregressive 6-step forecast — steps +1 and +2 most reliable, uncertainty grows beyond +3 |

---

## Requirements

```bash
pip install duckdb geopandas pandas numpy matplotlib seaborn
pip install rapidfuzz scikit-learn torch streamlit
pip install scipy Pillow
```

---

## How to Run

```bash
# 1. Preprocessing
python preprocessing.py --db database/aadhar.duckdb

# 2. Build adjacency matrix + tensor
python weighted_adjacency_matrix.py \
  --shp Adjacency_marix/2011_Dist.shp \
  --db  database/aadhar.duckdb

# 3. Clustering
cd Insights
python clustering.py --db ../database/aadhar.duckdb \
  --shp ../Adjacency_marix/2011_Dist.shp

# 4. Analysis scripts
python state_comparison.py --db ../database/aadhar.duckdb
python time_series.py      --db ../database/aadhar.duckdb
python table_analysis.py   --db ../database/aadhar.duckdb
python spatial_autocorr.py \
  --db  ../database/aadhar.duckdb \
  --shp ../Adjacency_marix/2011_Dist.shp \
  --w   ../adjacency_output/W_combined.csv

# 5. Train STGCN
cd ../STCGN
python stgcn_train.py

# 6. Forecast
python stgcn_forecast.py --steps 6

# 7. Launch dashboard
cd ../Insights
streamlit run app.py
```
