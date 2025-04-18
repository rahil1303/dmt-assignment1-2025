import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def advanced_preprocess(input_path="final_cleaned_dataset_filtered.csv",
                        output_path="advanced_preprocessed_dataset.csv"):

    df = pd.read_csv(input_path)

    # 🚧 Drop rows with missing target just in case
    df = df.dropna(subset=['ROOM_ESTIMATE_CLEANED', 'STRESS_LEVEL_CLEANED'])

    # 🔁 Feature Engineering
    df['study_pressure'] = df['ML_COURSE_NUM'] + df['DB_COURSE_NUM'] + df['IR_COURSE_NUM']
    df['sports_x_chatgpt'] = df['SPORTS_HOURS_CLEANED'] * df['CHATGPT_USAGE_NUM']
    df['sports_squared'] = df['SPORTS_HOURS_CLEANED'] ** 2
    df['chatgpt_squared'] = df['CHATGPT_USAGE_NUM'] ** 2
    df['age_month_product'] = df['AGE'] * df['BIRTHDAY_MONTH']

    # ✂️ Remove known outliers in ROOM_ESTIMATE_CLEANED
    df = df[(df['ROOM_ESTIMATE_CLEANED'] > 5) & (df['ROOM_ESTIMATE_CLEANED'] < 1500)]

    # 🪓 Winsorize ROOM_ESTIMATE_CLEANED for extreme values
    df['ROOM_ESTIMATE_WINSORIZED'] = winsorize(df['ROOM_ESTIMATE_CLEANED'], limits=[0.01, 0.01])

    # 🔁 Log Transform
    df['ROOM_ESTIMATE_LOG'] = np.log1p(df['ROOM_ESTIMATE_CLEANED'])

    # 🧮 Sqrt Transform
    df['ROOM_ESTIMATE_SQRT'] = np.sqrt(df['ROOM_ESTIMATE_CLEANED'])

    # ✅ Save it
    df.to_csv(output_path, index=False)
    print(f"✅ Advanced preprocessing complete! Output saved to: {output_path}")
    print(f"Rows retained: {df.shape[0]}")

# Optional: Run directly
if __name__ == "__main__":
    advanced_preprocess()
