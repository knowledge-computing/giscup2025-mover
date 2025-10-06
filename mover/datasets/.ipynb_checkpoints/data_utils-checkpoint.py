import numpy as np
import pandas as pd

from datasets.builtin import *


def load_traj_data(file):
    if '.csv.gz' in file:
        return pd.read_csv(file, compression='gzip')
    else:
        return pd.read_csv(file)


def load_location_profiles():
    LOC_PROFILES = {}
    for city_code in [1,2,3,4]:
        if city_code == 1:
            file = "datasets/location_profile/v2/cityA_norm.csv"
        elif city_code == 2:
            file = "datasets/location_profile/v2/cityB_norm.csv"
        elif city_code == 3:
            file = "datasets/location_profile/v2/cityC_norm.csv"
        elif city_code == 4:
            file = "datasets/location_profile/v2/cityD_norm.csv"
    
        loc_profiles = {(XY_SEP_TOKEN, XY_SEP_TOKEN): np.zeros(128)}
        df = pd.read_csv(file)
        cols = [col for col in df.columns if col != 'x' and col != 'y']
        print(f"... Processing location profiles in city `{city_code}`")
        for i, row in df.iterrows():
            loc_profiles[(row['x'], row['y'])] = np.array(list(row[cols]), dtype=np.float32)
    
        LOC_PROFILES[city_code] = loc_profiles
    return LOC_PROFILES


def load_user_profiles():
    USER_PROFILES = {}
    for city_code in [1,2,3,4]:
        if city_code == 1:
            file = "dataset/user_profile/cityA_user_embs_norm.csv"
        elif city_code == 2:
            file = "dataset/user_profile/cityB_user_embs_norm.csv"
        elif city_code == 3:
            file = "dataset/user_profile/cityC_user_embs_norm.csv"
        elif city_code == 4:
            file = "dataset/user_profile/cityD_user_embs_norm.csv"
    
        user_profiles = {}
        df = pd.read_csv(file)
        cols = [col for col in df.columns if col != 'uid']
        print(f"... Processing user profiles in city `{city_code}`")
        for i, row in df.iterrows():
            user_profiles[(row['uid'])] = np.array(list(row[cols]), dtype=np.float32)
    
        USER_PROFILES[city_code] = user_profiles
    return USER_PROFILES


def load_user_clusters():
    USER_CLUSTERS = {}
    for city_name in ['cityA', 'cityB', 'cityC', 'cityD']:
        file = f"datasets/user_cluster/{city_name}.csv"    
        user_cluster = {}
        df = pd.read_csv(file)
        print(f"... Processing user cluster in city `{city_name}`")
        for i, row in df.iterrows():
            user_cluster[row['uid']] = row['cluster']
        
        USER_CLUSTERS[city_name] = user_cluster
    return USER_CLUSTERS


def last_valid_token(tokens, fallback=0):
    valid = (tokens != XY_SEP_TOKEN) & (tokens != PAD_TOKEN)

    if not valid.any():
        return fallback

    idx = np.where(valid)[0][-1]
    return tokens[idx]




