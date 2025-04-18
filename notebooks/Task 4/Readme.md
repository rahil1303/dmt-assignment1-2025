# 🧠 Task 4 – Numerical Prediction (Regression)

This folder contains all code and data related to **Task 4** from the Data Mining Techniques course assignment.

---

## 📊 Datasets

We worked with **two main types of datasets** in this task:

### 🔹 1. `final_cleaned_dataset_filtered.csv`
- The **basic dataset** after manual cleaning and filtering.
- Contains original features such as course counts, chatGPT usage, sports hours, birthday info, etc.
- Used as the starting point for initial regression models.
- Target columns:
  - `STRESS_LEVEL_CLEANED`
  - `ROOM_ESTIMATE_CLEANED`

### 🔹 2. `copula_synthetic_dataset.csv`
- A **synthetically enhanced dataset** generated using advanced preprocessing + copula-based techniques.
- Aimed to handle data imbalance and enrich model training.
- Includes additional feature engineering columns like:
  - `study_pressure` (derived)
  - `sports_x_chatgpt` (interaction term)
- Used for second-phase experiments to observe model improvements.

Both datasets are aligned in terms of schema but differ in how they were constructed — giving us a solid foundation to test model performance before and after enhancement.

---

## 🎯 Targets

- `STRESS_LEVEL_CLEANED`: A continuous variable representing perceived stress.
- `ROOM_ESTIMATE_CLEANED` or `ROOM_ESTIMATE_LOG`: Log-transformed estimate of number of students in the room.

---

## 🛠️ Models Used

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

---

## 📈 Key Observations

- **Baseline models** (on the basic dataset) showed decent but limited performance.
- After **synthetic data generation** and feature enrichment, **Random Forest** showed a noticeable improvement — especially in MAE and MSE.
- We also used **individual case predictions** (sample index 3) to compare how each model performed in real terms.

---

## 🗂️ Scripts Overview

| Script | Description |
|--------|-------------|
| `evaluation.py` | Handles cross-validation, logging, and metric output |
| `preprocess.py` | Prepares data, handles missing values, scaling |
| `*_cases.py` | Case-specific prediction comparisons across models |
| `visualize_results.py` | Generates plots for MAE/MSE with error bars and scatter analysis |

---

## 📍 Directory

This folder lives under:  
`notebooks/Task 4`

---

**Author:** Rahil Sharma  
**Course:** Data Mining Techniques (2025)
