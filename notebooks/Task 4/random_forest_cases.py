# random_forest_cases.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
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

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(df[feature_cols])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    df = df[df['STRESS_LEVEL_CLEANED'].notna() & df['ROOM_ESTIMATE_CLEANED'].notna()]

    return X_scaled, df

def evaluate_model(model, X, y, target_name):
    mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    mae_scores = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
    print(f"\nRandom Forest Regressor on {target_name}:")
    print(f"MSE: {mse_scores.mean():.2f} ± {mse_scores.std():.2f}")
    print(f"MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")

def run_case1_random_forest():
    X, df = load_dataset("/content/drive/MyDrive/Data Mining Techniques/Task-4/advanced_preprocessed_dataset.csv")
    y = df['STRESS_LEVEL_CLEANED']
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    evaluate_model(model, X, y, target_name="STRESS_LEVEL_CLEANED")

def run_case2_random_forest():
    X, df = load_dataset("/content/drive/MyDrive/Data Mining Techniques/Task-4/advanced_preprocessed_dataset.csv")
    y = df['ROOM_ESTIMATE_CLEANED']
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    evaluate_model(model, X, y, target_name="ROOM_ESTIMATE_CLEANED")

def run_case3_random_forest():
    X, df = load_dataset("/content/drive/MyDrive/Data Mining Techniques/Task-4/advanced_preprocessed_dataset.csv")
    Y = df[['STRESS_LEVEL_CLEANED', 'ROOM_ESTIMATE_CLEANED']]

    print("\nMulti-Output Random Forest (Case 3):")
    for target in Y.columns:
        y = Y[target]
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        evaluate_model(model, X, y, target_name=target)

def run_case3_random_forest_mo():
    X, df = load_dataset("/content/drive/MyDrive/Data Mining Techniques/Task-4/advanced_preprocessed_dataset.csv")
    Y = df[['STRESS_LEVEL_CLEANED', 'ROOM_ESTIMATE_CLEANED']]

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
    model.fit(X, Y)
    y_pred = model.predict(X)

    print("\nMulti-Output Random Forest (Case 3 using MultiOutputRegressor):")
    for i, target in enumerate(Y.columns):
        mse = np.mean((Y.iloc[:, i] - y_pred[:, i]) ** 2)
        mae = np.mean(np.abs(Y.iloc[:, i] - y_pred[:, i]))
        print(f"{target}: MSE = {mse:.2f}, MAE = {mae:.2f}")

def run_case2_random_forest_log():
    X, df = load_dataset("/content/drive/MyDrive/Data Mining Techniques/Task-4/advanced_preprocessed_dataset.csv")
    y = df['ROOM_ESTIMATE_LOG']
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    evaluate_model(model, X, y, target_name="ROOM_ESTIMATE_LOG")

def run_case3_random_forest_log():
    X, df = load_dataset("/content/drive/MyDrive/Data Mining Techniques/Task-4/advanced_preprocessed_dataset.csv")
    Y = df[['STRESS_LEVEL_CLEANED', 'ROOM_ESTIMATE_LOG']]

    print("\nMulti-Output Random Forest (Case 3: STRESS + LOG ROOM ESTIMATE):")
    for target in Y.columns:
        y = Y[target]
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        evaluate_model(model, X, y, target_name=target)

def evaluate_model_with_training(model, X, y, target_name=""):
    model.fit(X, y)
    y_train_pred = model.predict(X)
    train_mse = np.mean((y - y_train_pred) ** 2)
    train_mae = np.mean(np.abs(y - y_train_pred))

    cv_mse = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_mae = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)

    print(f"\n{model.__class__.__name__} on {target_name}:")
    print(f"Training MSE: {train_mse:.2f}, MAE: {train_mae:.2f}")
    print(f"CV MSE: {cv_mse.mean():.2f} ± {cv_mse.std():.2f}")
    print(f"CV MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")

def run_random_forest_training_vs_cv():
    X, df = load_dataset("/content/drive/MyDrive/Data Mining Techniques/Task-4/advanced_preprocessed_dataset.csv")
    print("\n[Log Target] Training vs CV Error Check")
    evaluate_model_with_training(RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42), X, df['ROOM_ESTIMATE_LOG'], target_name="ROOM_ESTIMATE_LOG")
    print("\n[Clean Target] Training vs CV Error Check")
    evaluate_model_with_training(RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42), X, df['ROOM_ESTIMATE_CLEANED'], target_name="ROOM_ESTIMATE_CLEANED")
