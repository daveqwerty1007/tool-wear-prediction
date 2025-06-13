import pandas as pd
import os

def load_vicomtech_data(raw_path='../data/raw/VicomtechToolWearData.csv', selected_only=False):

    """
    Load the Vicomtech dataset and perform basic cleaning. Columns are
    renamed for clarity and rows without a flank wear label are dropped.

    Parameters:
        raw_path (str): Path to the CSV dataset
        selected_only (bool): If True, return only selected columns useful for modeling

    Returns:
        pd.DataFrame
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip()


    # Drop rows with no flank wear label
    if 'vb' in df.columns:
        df = df.dropna(subset=['vb'])

    
    return df


if __name__ == "__main__":
    df = load_vicomtech_data()
    print("✅ Data loaded:", df.shape)

