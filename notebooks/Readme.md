## ✅ Project Progress Summary (As of April 2025)

This repository presents the completed first iteration of Assignment 1 for the Data Mining Techniques course (2025). The following components have been successfully implemented:

### 🔍 Exploratory Data Analysis (EDA)
- Cleaned and parsed the ODI-2025 survey dataset
- Visualized distributions of stress and sports activity
- Handled non-numeric anomalies (e.g., "Over 9000", "4-5")
- Grouped and compared sports hours across gender identities

### 🤖 Classification Models
- **Target**: Binned stress level (low, medium, high)
- **Baseline model**: Sports hours as sole feature (Naive Bayes, Decision Tree)
- **Expanded model**: Included ChatGPT usage, bedtime, ML background
- Confusion matrices plotted for both versions
- Accuracy evaluated and compared (~36–44%)

### 📈 Regression Model
- **Target**: Numeric stress score (0–100)
- Applied Linear Regression on engineered features
- Handled outliers and data type inconsistencies
- RMSE ≈ 30.9, R² ≈ –0.03  
  → Indicates model predicts near the mean; limited explanatory power

### 🔗 Association Rule Mining
- Converted selected features into binary format
- Used Apriori algorithm to mine frequent patterns
- Extracted rules with high confidence (≥ 0.9)
- Plotted support vs. confidence with lift-based scaling
- Observed moderate links between ChatGPT usage, late bedtimes, and ML background

---

### 🧠 Potential Future Improvements

- Perform **feature selection** or **dimensionality reduction** (e.g., PCA)
- Try **Random Forest** or **SVM** for classification
- Engineer new features like:
  - Study hours
  - Time spent on assignments
  - Interaction terms (e.g., sports × bedtime)
- Expand association rules with higher granularity
- Conduct cross-validation for more robust evaluation
- Improve interpretability (e.g., SHAP or decision path tracing)

---

This iteration provides a strong foundation and a reproducible pipeline. Future work will focus on refinement, feature engineering, and potentially ensemble models.

