import pandas as pd
import os

def load_vicomtech_data(raw_path='data/raw/VicomtechToolWearData.csv', selected_only=True):
    """
    Load and clean the Vicomtech tool wear dataset with renamed columns.

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

    # Rename columns for clarity
    rename_map = {col: col.lower().replace('.', '_').replace('__', '_') for col in df.columns}
    manual_renames = {
        "ae_rms": "acoustic_rms",
        "ae_max": "acoustic_peak",
        "vb": "flank_wear",
        "tool": "tool_id",
        "f_c_rms": "cutting_force_rms",
        "f_c_max": "cutting_force_max"
    }
    rename_map.update(manual_renames)
    df.rename(columns=rename_map, inplace=True)

    # Drop rows with no flank wear label
    if 'flank_wear' in df.columns:
        df = df.dropna(subset=['flank_wear'])

    # Optionally select only modeling-relevant features
    if selected_only:
        selected_cols = [
            'tool_id', 'acoustic_rms', 'acoustic_peak', 'cutting_force_rms',
            'cutting_force_max', 'flank_wear'
        ]
        df = df[[col for col in selected_cols if col in df.columns]]

    return df


if __name__ == "__main__":
    df = load_vicomtech_data()
    print("✅ Data loaded:", df.shape)
    print(df.head())
