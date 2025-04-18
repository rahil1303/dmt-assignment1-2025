# visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def extract_model_results_from_logs(log_path):
    results = []
    with open(log_path, 'r') as file:
        current_model = None
        current_target = None

        for line in file:
            line = line.strip()

            # Match model and target line
            match_model = re.match(r"^(.*) on (.*):$", line)
            if match_model:
                current_model = match_model.group(1)
                current_target = match_model.group(2)

            # Match MSE/MAE line
            match_mse = re.match(r"MSE: ([0-9.]+) ± ([0-9.]+)", line)
            if match_mse:
                mse = float(match_mse.group(1))
                mse_std = float(match_mse.group(2))

            match_mae = re.match(r"MAE: ([0-9.]+) ± ([0-9.]+)", line)
            if match_mae:
                mae = float(match_mae.group(1))
                mae_std = float(match_mae.group(2))

                # Only save if all parts are available
                if current_model and current_target:
                    results.append((current_model, current_target, mse, mse_std, mae, mae_std))
                    current_model = None
                    current_target = None

    return pd.DataFrame(results, columns=["Model", "Target", "MSE", "MSE_std", "MAE", "MAE_std"])

def plot_results(df_vis):
    # Plot 1: MAE by model and target
    plt.figure(figsize=(10, 5))
    for target in df_vis["Target"].unique():
        subset = df_vis[df_vis["Target"] == target]
        plt.errorbar(subset["Model"], subset["MAE"], yerr=subset["MAE_std"], label=target, capsize=4, marker='o')
    plt.title("MAE by Model and Target Variable")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Plot 2: MSE by model and target
    plt.figure(figsize=(10, 5))
    for target in df_vis["Target"].unique():
        subset = df_vis[df_vis["Target"] == target]
        plt.errorbar(subset["Model"], subset["MSE"], yerr=subset["MSE_std"], label=target, capsize=4, marker='s')
    plt.title("MSE by Model and Target Variable")
    plt.ylabel("Mean Squared Error")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Plot 3: MAE vs MSE (for all models and both targets)
    plt.figure(figsize=(8, 6))
    for _, row in df_vis.iterrows():
        plt.scatter(row["MAE"], row["MSE"], label=f"{row['Model']} ({row['Target']})")
    plt.xlabel("MAE")
    plt.ylabel("MSE")
    plt.title("MAE vs MSE Across Models")
    plt.legend(fontsize='small', loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage in notebook:
# df = extract_model_results_from_logs("model_output_log.txt")
# plot_results(df)
