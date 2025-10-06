import pandas as pd
import numpy  as np

def min_max_normalize_ignore_zeros(df):
    df_norm = df.copy()
    for col in df.columns:
        if col == 'x' or col == 'y' or col == 'uid':
            continue
        
        col_data = df[col].values.astype(float)
        mask = col_data != 0  # mask out zeros

        if np.any(mask):  # skip if all values are zero
            col_min = col_data[mask].min()
            col_max = col_data[mask].max()

            df_norm[col] = df_norm[col].astype(float)
            if col_max != col_min:
                df_norm.loc[mask, col] = (col_data[mask] - col_min) / (col_max - col_min)
            else:
                df_norm.loc[mask, col] = col_min  

    return df_norm


profile_file = "location_profile/v2/cityD.csv"
loc_profile_dict = {}
df = pd.read_csv(profile_file)
                
profile = min_max_normalize_ignore_zeros(df)
profile.to_csv("location_profile/v2/cityD_norm.csv", index=False)
