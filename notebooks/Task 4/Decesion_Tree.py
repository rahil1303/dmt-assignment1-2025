# decision_tree_cases.py
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

def load_dataset(path):
    df = pd.read_csv(path)

    # Feature engineering
    df['study_pressure'] = df['ML_COURSE_NUM'] + df['DB_COURSE_NUM'] + df['IR_COURSE_NUM']
    df['sports_x_chatgpt'] = df['SPORTS_HOURS_CLEANED'] * df['CHATGPT_USAGE_NUM']

    feature_cols = [
        'SPORTS_HOURS_CLEANED', 'CHATGPT_USAGE_NUM',
        'ML_COURSE_NUM', 'DB_COURSE_NUM', 'IR_COURSE_NUM',
        'AGE', 'BIRTHDAY_MONTH', 'BIRTHDAY_DAY',
        'study_pressure', 'sports_x_chatgpt'
    ]

    # Impute missing values in features
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(df[feature_cols])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Drop only rows with missing targets
    df = df[df['STRESS_LEVEL_CLEANED'].notna() & df['ROOM_ESTIMATE_CLEANED'].notna()]

    return X_scaled, df

def evaluate_model(model, X, y, target_name):
    mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    mae_scores = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
    print(f"\nDecision Tree Regressor on {target_name}:")
    print(f"MSE: {mse_scores.mean():.2f} ± {mse_scores.std():.2f}")
    print(f"MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")

def run_case1_decision_tree():
    X, df = load_dataset("/content/drive/MyDrive/Data Mining Techniques/Task-4/final_cleaned_dataset_filtered.csv")
    y = df['STRESS_LEVEL_CLEANED']

    configs = [
        {"max_depth": 4, "min_samples_leaf": 15},
        {"max_depth": 6, "min_samples_leaf": 5},
        {"max_depth": 8, "min_samples_leaf": 2},
    ]

    for cfg in configs:
        print(f"\nTuning Config: max_depth={cfg['max_depth']}, min_samples_leaf={cfg['min_samples_leaf']}")
        model = DecisionTreeRegressor(max_depth=cfg['max_depth'], min_samples_leaf=cfg['min_samples_leaf'], random_state=42)
        evaluate_model(model, X, y, target_name="STRESS_LEVEL_CLEANED")

def run_case2_decision_tree():
    X, df = load_dataset("/content/drive/MyDrive/Data Mining Techniques/Task-4/final_cleaned_dataset_filtered.csv")
    y = df['ROOM_ESTIMATE_CLEANED']

    configs = [
        {"max_depth": 4, "min_samples_leaf": 15},
        {"max_depth": 6, "min_samples_leaf": 5},
        {"max_depth": 8, "min_samples_leaf": 2},
    ]

    for cfg in configs:
        print(f"\nTuning Config: max_depth={cfg['max_depth']}, min_samples_leaf={cfg['min_samples_leaf']}")
        model = DecisionTreeRegressor(max_depth=cfg['max_depth'], min_samples_leaf=cfg['min_samples_leaf'], random_state=42)
        evaluate_model(model, X, y, target_name="ROOM_ESTIMATE_CLEANED")

def run_case3_decision_tree():
    X, df = load_dataset("/content/drive/MyDrive/Data Mining Techniques/Task-4/final_cleaned_dataset_filtered.csv")
    Y = df[['STRESS_LEVEL_CLEANED', 'ROOM_ESTIMATE_CLEANED']]

    configs = [
        {"max_depth": 4, "min_samples_leaf": 15},
        {"max_depth": 6, "min_samples_leaf": 5},
        {"max_depth": 8, "min_samples_leaf": 2},
    ]

    print("\nMulti-Output Decision Tree (Case 3):")
    for cfg in configs:
        print(f"\nTuning Config: max_depth={cfg['max_depth']}, min_samples_leaf={cfg['min_samples_leaf']}")
        for target in Y.columns:
            y = Y[target]
            model = DecisionTreeRegressor(max_depth=cfg['max_depth'], min_samples_leaf=cfg['min_samples_leaf'], random_state=42)
            evaluate_model(model, X, y, target_name=target)
