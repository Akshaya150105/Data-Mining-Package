# STGCN for Aadhaar Data (Biometric & Enrolment Forecasting)

## 📌 Overview

This project implements a **Spatio-Temporal Graph Convolutional Network (STGCN)** pipeline to forecast Aadhaar-related metrics across Indian districts.

The model captures:

* **Spatial relationships** between districts (graph)
* **Temporal dynamics** in weekly data (time series)

We build **two separate models**:

* 📊 **Biometric Model** → predicts `bio_total`
* 📈 **Enrolment Model** → predicts `enrol_total`

---

## 🧠 Key Idea

* Each **district = node**
* Graph edges represent:

  * geographic proximity (distance)
  * behavioral similarity (feature similarity)
* Weekly data forms a tensor:

  ```
  X shape = [T, N, C]
  T = weeks
  N = districts
  C = features
  ```

STGCN learns:

* temporal patterns (weekly trends)
* spatial interactions (district influence)

---

## 📂 Project Structure

```
NEW_STGCN/
│
├── build_district_graph.py
├── build_biometric_weekly_tensor.py
├── build_enrolment_weekly_tensor.py
├── stgcn_train_single_target.py
│
├── graph_output/
├── biometric_weekly_output/
├── enrolment_weekly_output/
├── biometric_model_output/
├── enrolment_model_output/
│
├── Adjacency_marix/
│   └── 2011_Dist.shp
│
└── database/
    └── aadhar.duckdb
```

---

## ⚙️ Installation

```bash
pip install geopandas duckdb numpy pandas scipy matplotlib seaborn rapidfuzz scikit-learn torch
```

---

## 🚀 Full Pipeline

---

### 🔹 Step 1 — Build District Graph

```bash
python build_district_graph.py --shp ..\Adjacency_marix\2011_Dist.shp --db ..\database\aadhar.duckdb
```

### Output (`graph_output/`)

* `L_normalised_laplacian.csv`  
* `district_order.csv` 
* `W_distance.csv`, `W_similarity.csv`, `W_combined.csv`

---

### 🔹 Step 2 — Build Biometric Weekly Tensor

```bash
python build_biometric_weekly_tensor.py
```

### Output (`biometric_weekly_output/`)

* `feature_tensor_X.npy`
* `tensor_feature_columns.csv`
* `week_index.csv`

---

### 🔹 Step 3 — Build Enrolment Weekly Tensor

```bash
python build_enrolment_weekly_tensor.py
```

### Output (`enrolment_weekly_output/`)

* `feature_tensor_X.npy`
* `tensor_feature_columns.csv`
* `week_index.csv`

---

### 🔹 Step 4 — Train Biometric Model

```bash
python stgcn_train_single_target.py ^
  --tensor biometric_weekly_output/feature_tensor_X.npy ^
  --laplacian graph_output/L_normalised_laplacian.csv ^
  --districts graph_output/district_order.csv ^
  --week_index biometric_weekly_output/week_index.csv ^
  --features biometric_weekly_output/tensor_feature_columns.csv ^
  --target bio_total ^
  --output_dir biometric_model_output
```

---

### 🔹 Step 5 — Train Enrolment Model

```bash
python stgcn_train_single_target.py ^
  --tensor enrolment_weekly_output/feature_tensor_X.npy ^
  --laplacian graph_output/L_normalised_laplacian.csv ^
  --districts graph_output/district_order.csv ^
  --week_index enrolment_weekly_output/week_index.csv ^
  --features enrolment_weekly_output/tensor_feature_columns.csv ^
  --target enrol_total ^
  --output_dir enrolment_model_output
```

---

## 📊 Outputs

### 📁 Model Output Folder

Each model produces:

* `predictions_by_district_week.csv` ⭐
* `district_metrics.csv`
* `metrics.txt`
* `best_model.pt`

---

### 📄 Example CSV

| week_start | district | actual | predicted | abs_error |
| ---------- | -------- | ------ | --------- | --------- |
| 2024-03-04 | Chennai  | 25011  | 24680     | 330       |
| 2024-03-04 | Madurai  | 11200  | 10980     | 220       |

---

## 🔗 How Everything Connects

```
district_order.csv
        ↓
Graph (Laplacian)
        ↓
Tensor [T, N, C]
        ↓
STGCN Model
        ↓
Predictions
```

👉 **Critical Rule**:
The district order must be **identical** in:

* graph
* tensor
* model

---

## 🧪 Features Used

### Biometric

* bio_total
* bio_age_5_17
* bio_age_17_
* age_5_ratio
* age_17_ratio
* dependency_ratio

### Enrolment

* enrol_total
* age_0_5
* age_5_17
* age_18_greater
* enrol_minor_ratio
* enrol_adult_ratio

---

## ⚠️ Important Design Choices

### ✅ Weekly Aggregation

* fixes irregular date spacing
* ensures equal temporal intervals

### ✅ No Data Leakage

* scalers fitted **only on training data**

### ✅ Separate Models

* biometric and enrolment trained independently

### ✅ Combined Graph

* distance + similarity

---

## 🔮 Future Improvements

* Add time features (month, seasonality)
* Multi-step forecasting (`t_out > 1`)
* Visualization (map-based predictions)

---

## 🧠 Summary

This project builds a **graph-based forecasting system** where:

* Graph = district relationships
* Tensor = weekly Aadhaar data
* STGCN = learns spatial + temporal patterns

---

## 👨‍💻 Author Notes

* Designed for Aadhaar district-level analysis
* Optimized for interpretability and modularity
* Supports easy experimentation with graph types and features



