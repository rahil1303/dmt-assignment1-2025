# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_path, target_column):
    df = pd.read_csv(csv_path)

    # Add engineered features
    df['study_pressure'] = df['ML_COURSE_NUM'] + df['DB_COURSE_NUM'] + df['IR_COURSE_NUM']
    df['sports_x_chatgpt'] = df['SPORTS_HOURS_CLEANED'] * df['CHATGPT_USAGE_NUM']

    feature_cols = [
        'SPORTS_HOURS_CLEANED', 'CHATGPT_USAGE_NUM',
        'ML_COURSE_NUM', 'DB_COURSE_NUM', 'IR_COURSE_NUM',
        'AGE', 'BIRTHDAY_MONTH', 'BIRTHDAY_DAY',
        'study_pressure', 'sports_x_chatgpt'
    ]

    df = df[feature_cols + [target_column]].dropna()

    X = df[feature_cols]
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values