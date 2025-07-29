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
    # remove leading/trailing spaces from column headers
    df.columns = df.columns.str.strip()

    # Rename common columns for easier access
    rename_map = {
        'Tool': 'tool_id',
        'ToolID': 'tool_id',
        'AE_RMS': 'acoustic_rms',
        'AE_MAX': 'acoustic_peak',
        'F_c_RMS': 'cutting_force_rms',
        'F_c_MAX': 'cutting_force_max',
        'Vb': 'flank_wear'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # normalize column names to lower case for consistency
    df.columns = df.columns.str.lower()

    # Drop rows with no flank wear label
    if 'flank_wear' in df.columns:
        df = df.dropna(subset=['flank_wear'])

    # Create simple features if the required columns are present
    if {'cutting_force_max', 'cutting_force_rms'}.issubset(df.columns):
        df['force_ratio'] = df['cutting_force_max'] / df['cutting_force_rms']
        df['force_diff'] = df['cutting_force_max'] - df['cutting_force_rms']
    if {'acoustic_peak', 'acoustic_rms'}.issubset(df.columns):
        df['acoustic_mean'] = df[['acoustic_rms', 'acoustic_peak']].mean(axis=1)
        df['acoustic_diff'] = df['acoustic_peak'] - df['acoustic_rms']
    if {'acoustic_peak', 'cutting_force_max'}.issubset(df.columns):
        df['acoustic_to_force'] = df['acoustic_peak'] / df['cutting_force_max']

    selected_cols = [
        'tool_id',
        'acoustic_rms',
        'acoustic_peak',
        'cutting_force_rms',
        'cutting_force_max',
        'force_ratio',
        'force_diff',
        'acoustic_mean',
        'acoustic_diff',
        'acoustic_to_force',
        'flank_wear'
    ]

    if selected_only:
        return df[[c for c in selected_cols if c in df.columns]]
    return df


if __name__ == "__main__":
    df = load_vicomtech_data()
    print("✅ Data loaded:", df.shape)

