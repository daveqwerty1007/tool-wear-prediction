import pandas as pd
import os
def load_vicomtech_data(raw_path = 'data/raw/vicomtech/VicomtechToolWearData.csv'):
    '''
    load and some cleaning on vicomtech data set
    '''
    if not os.path.exists(raw_path):
        raise FileNotFoundError
    df = pd.read_csv(raw_path)
    
    df.columns = df.colums.str.strip() #remove space
    df = df.dropna(subset = ['Vb']) # remove rows without lables
    
    selected_cols = [
        'ToolID', 'Segment', 'Vb', 
        'Fz_mean', 'Fz_max', 
        'AE_RMS', 'AE_Peak', 
        'F_c_mean', 'F_c_max',
        'Speed', 'Torque'
    ]
    
    df = df[[col for col in selected_cols if col in df.columns]]
    return df

if __name__ == "__main__":
    df = load_vicomtech_data()
    print("✅ Data loaded:", df.shape)
    print(df.head()) 