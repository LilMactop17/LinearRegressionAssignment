import os
import numpy as np
import pandas as pd

INPUT_PATH = "src/Dataset.xlsx"
OUTPUT_PATH = "DatasetSorted.xlsx"

def sort_voltage_cells_per_row(df):
    voltage_cols = [f"U{i}" for i in range(1, 21)]
    for idx in df.index:
        values = df.loc[idx, voltage_cols].to_numpy(dtype=float)
        values_sorted = np.sort(values)
        df.loc[idx, voltage_cols] = values_sorted
    return df

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"{INPUT_PATH} not found")

    df = pd.read_excel(INPUT_PATH)
    df_sorted = sort_voltage_cells_per_row(df)
    df_sorted.to_excel(OUTPUT_PATH, index=False)
    print(f"Saved sorted dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
