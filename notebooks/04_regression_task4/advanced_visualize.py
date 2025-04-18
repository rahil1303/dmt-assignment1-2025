import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def extract_model_results_from_logs(log_path):
    results = []
    with open(log_path, 'r') as file:
        current_model = None
        current_target = None

        for line in file:
            line = line.strip()

            match_model = re.match(r"^(.*) on (.*):$", line)
            if match_model:
                current_model = match_model.group(1)
                current_target = match_model.group(2)

            match_mse = re.match(r"CV MSE: ([0-9.]+) ± ([0-9.]+)", line)
            if match_mse:
                mse = float(match_mse.group(1))
                mse_std = float(match_mse.group(2))

            match_mae = re.match(r"CV MAE: ([0-9.]+) ± ([0-9.]+)", line)
            if match_mae:
                mae = float(match_mae.group(1))
                mae_std = float(match_mae.group(2))

                if current_model and current_target:
                    results.append((current_model, current_target, mse, mse_std, mae, mae_std))
                    current_model = None
                    current_target = None

    return pd.DataFrame(results, columns=["Model", "Target", "MSE", "MSE_std", "MAE", "MAE_std"])

def plot_results(df_vis):
    # --- MAE with Error Bars ---
    plt.figure(figsize=(10, 5))
    for target in df_vis["Target"].unique():
        subset = df_vis[df_vis["Target"] == target]
        plt.errorbar(subset["Model"], subset["MAE"], yerr=subset["MAE_std"],
                     label=target, capsize=4, marker='o')
    plt.title("MAE by Model and Target Variable (Copula Synthetic)")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=30)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mae_plot.png', bbox_inches='tight')  # Save with tight bounding box
    plt.show()
    plt.close()

    # --- MSE with Error Bars ---
    plt.figure(figsize=(10, 5))
    for target in df_vis["Target"].unique():
        subset = df_vis[df_vis["Target"] == target]
        plt.errorbar(subset["Model"], subset["MSE"], yerr=subset["MSE_std"],
                     label=target, capsize=4, marker='s')
    plt.title("MSE by Model and Target Variable (Copula Synthetic)")
    plt.ylabel("Mean Squared Error")
    plt.xticks(rotation=30)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mse_plot.png', bbox_inches='tight')  # Save with tight bounding box
    plt.show()
    plt.close()

    # --- Split Scatterplots: One per Target ---
    targets = df_vis["Target"].unique()
    
    # Create figure with proper spacing
    fig, axes = plt.subplots(1, len(targets), figsize=(14, 6), constrained_layout=True)
    
    # Handle single target case
    if len(targets) == 1:
        axes = [axes]
    
    for i, target in enumerate(targets):
        subset = df_vis[df_vis["Target"] == target]
        ax = axes[i]
        sns.scatterplot(
            data=subset,
            x="MAE",
            y="MSE",
            hue="Model",
            s=120,
            ax=ax,
            edgecolor="black",
            linewidth=0.6
        )
        
        # Add model labels with a small offset
        for _, row in subset.iterrows():
            ax.annotate(
                row["Model"], 
                (row["MAE"], row["MSE"]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        ax.set_title(target)
        ax.set_xlabel("MAE")
        ax.set_ylabel("MSE")
        ax.grid(True)
        
        # Remove legend if it's cluttering the plot
        if i > 0:
            ax.set_ylabel("")  # Only keep ylabel on first subplot
    
    plt.suptitle("MAE vs MSE Across Models (Per Target)", fontsize=14)
    
    # Save with tight bounding box
    plt.savefig('mae_vs_mse_scatter.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

# Example usage:
# df = extract_model_results_from_logs('your_log_file.txt')
# plot_results(df)