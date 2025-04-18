# evaluation.py
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(model, X, y, target_name):
    # Train on full data
    model.fit(X, y)
    train_preds = model.predict(X)
    train_mse = np.mean((train_preds - y) ** 2)
    train_mae = np.mean(np.abs(train_preds - y))

    # Cross-validation
    mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    mae_scores = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)

    print(f"\nRandom Forest Regressor on {target_name}:")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"CV MSE: {mse_scores.mean():.2f} ± {mse_scores.std():.2f}")
    print(f"CV MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
