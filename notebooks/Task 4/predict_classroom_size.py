# predict_classroom_size.py
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_and_preprocess
from evaluation import evaluate_model

def predict_classroom_size():
    X, y = load_and_preprocess("/content/drive/MyDrive/Data Mining Techniques/Task-4/final_cleaned_dataset.csv", target_column="ROOM_ESTIMATE_CLEANED")

    linreg = LinearRegression()
    dtree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)

    print("Evaluating Linear Regression on classroom size:")
    evaluate_model(linreg, X, y, model_name="Linear Regression")

    print("\nEvaluating Decision Tree Regressor on classroom size:")
    evaluate_model(dtree, X, y, model_name="Decision Tree Regressor")

    print("\nEvaluating Random Forest Regressor on classroom size:")
    evaluate_model(rf, X, y, model_name="Random Forest Regressor")