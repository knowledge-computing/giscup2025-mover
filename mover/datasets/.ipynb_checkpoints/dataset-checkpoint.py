import json
import copy
import random
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from datasets.builtin import *
from datasets.data_utils import *

######################################################
######################################################
######################################################


def process_single_trajectory(city_name, traj):
    d = traj['d'].to_numpy() # [1, 75]
    t = traj['t'].to_numpy() + 1 # [0, 47] => [1, 48]
    input_x = copy.deepcopy(traj['x'].to_numpy())
    input_y = copy.deepcopy(traj['y'].to_numpy())
    time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - 
                           (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
    time_delta[time_delta > 47] = 48
    time_delta[0] = 48
    label_x = traj['x'].to_numpy() - 1
    label_y = traj['y'].to_numpy() - 1
    trip_id = traj['trip_id'].to_numpy()
    trip_seq = traj['seq_in_trip'].to_numpy()
    
    city = np.full(len(d), CITY2ID[city_name], dtype=int)
    
    dow = (d % 7 + 5) % 7
    if city_name in ['cityC', 'cityD']:
        dow[d > 60] = (dow[d > 60] + 5) % 7
    dow[dow == 0] = 7 # [1, 7]

    sep = traj['trip_id'] == '0_0_0'
    d[sep] = DAY_SEP_TOKEN
    t[sep] = TIME_SEP_TOKEN
    dow[sep] = DOW_SEP_TOKEN
    city[sep] = CITY_SEP_TOKEN
    input_x[sep] = XY_SEP_TOKEN
    input_y[sep] = XY_SEP_TOKEN
    time_delta[sep] = TIME_SEP_TOKEN
    label_x[sep] = XY_SEP_TOKEN
    label_y[sep] = XY_SEP_TOKEN
    trip_seq[sep] = TIME_SEP_TOKEN

    return {
        'd': d,
        't': t,
        'dow': dow,
        'city': city,
        'input_x': input_x,
        'input_y': input_y,
        'time_delta': time_delta,
        'label_x': label_x,
        'label_y': label_y,
        'trip_id': trip_id,
        'trip_seq': trip_seq,
    }


class BaseDataset(Dataset):
    def __init__(self, cities, config):
        super().__init__()
        self.uid_array = []
        self.d_array = []
        self.t_array = []
        self.dow_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.city_array = []
        self.trip_id_array = []
        self.trip_seq_array = []
        
        self.config = config
        if not isinstance(cities, list):
            cities = [cities]
        self.cities = cities
        

LOC_PROFILES = load_location_profiles()
USER_CLUSTERS = load_user_clusters()

class PretrainDataset(BaseDataset):
    def __init__(self, cities, config, is_train=True):
        super().__init__(cities, config)
        self.is_train = is_train
        self.mask_type = config.DATA.MASK_TYPE
        self.max_seq_len = config.MODEL.MOBERT.ENCODER.MAX_SEQ_LEN
        
        for city_name in self.cities:
            print(f"=> Loading data in {city_name}")
            traj_df = load_traj_data(CITY2PATH[city_name])

            user_ids = pd.unique(traj_df['uid'])
            num_users = user_ids.shape[0]
            num_val, num_test = 3000, 3000
            val_user_ids = user_ids[-(num_test + num_val): -num_test]
            test_user_ids = user_ids[-num_test:]
            
            if self.is_train:
                mask = (traj_df['uid'].isin(val_user_ids)) & (traj_df['d'] > 60)
                if city_name == "cityA":
                    mask |= (traj_df['uid'].isin(user_ids[:90000]))
                    
                traj_df.loc[mask, ['x', 'y']] = 999
                traj_df = traj_df[(traj_df['x'] < 999) & (traj_df['y'] < 999)]
            else:
                traj_df = traj_df[traj_df['uid'].isin(val_user_ids)]
                
            for uid, traj in tqdm(traj_df.groupby('uid')):
                traj_out = process_single_trajectory(city_name, traj)
                self.d_array.append(traj_out['d'])  # 1-75; 0:<pad>
                self.t_array.append(traj_out['t'])  # 1-48; 0:<pad>
                self.dow_array.append(traj_out['dow']) # 1-7; 0:<pad>
                self.city_array.append(traj_out['city'])   # add: city ABCD encode 1234
                self.input_x_array.append(traj_out['input_x'])  # 1-200; 0:<pad>; 201:<mask>
                self.input_y_array.append(traj_out['input_y'])  # 1-200; 0:<pad>; 201:<mask>
                self.time_delta_array.append(traj_out['time_delta']) # 1-48; 0:<pad>
                self.label_x_array.append(traj_out['label_x'])  # 0-199
                self.label_y_array.append(traj_out['label_y'])  # 0-199
                self.trip_id_array.append(traj_out['trip_id'])
                self.trip_seq_array.append(traj_out['trip_seq'])
        
    def get_sub_traj(self, traj_len):
        if traj_len < self.max_seq_len:
            s = 0 
            e = traj_len
        elif self.is_train:
            s = random.randint(0, traj_len - self.max_seq_len)
            e = s + self.max_seq_len
        else:
            e = traj_len
            s = traj_len - self.max_seq_len
        return s, e

    def get_mask(self, d, t, dow, trip_id):
        d_unique = np.unique(d[d != DAY_SEP_TOKEN])
        L = d.shape[0]
        mask = np.zeros(L, dtype=bool)

        if self.is_train:
            mask_type = random.choice(self.mask_type)
        else:
            mask_type = "last_days"

        if mask_type == "continuous_days":
            d_min = np.min(d_unique)
            d_max = np.max(d_unique)
            num_mask_d = max(1, int(d_unique.size * 0.15))
            mask_start = np.random.choice(d_unique[d_unique <= d_max - num_mask_d])
            mask_end = mask_start + num_mask_d - 1
            mask |= (d >= mask_start) & (d <= mask_end)
            
        elif mask_type == "random_days":
            num_mask_d = max(1, int(d_unique.size * 0.15))
            mask_ds = np.random.choice(d_unique, size=num_mask_d, replace=False)
            mask |= np.isin(d, mask_ds)

        elif mask_type == "random_time":
            t_unique = np.unique(t[t != TIME_SEP_TOKEN])
            mask_t = np.random.choice(t_unique)
            mask |= (t == mask_t)
            
        elif mask_type == "random_dow":
            dow_unique = np.unique(dow[dow != DOW_SEP_TOKEN])
            mask_dow = np.random.choice(dow_unique)
            mask |= (t == mask_dow)

        elif mask_type == "random_trips":
            trip_unique = np.unique(trip_id[trip_id != '0_0_0'])
            num_mask_trip = max(1, int(round(trip_unique.size * 0.15)))
            mask_trip = np.random.choice(trip_unique, size=num_mask_trip, replace=False)
            mask |= np.isin(trip_id, mask_trip)

        elif mask_type == "last_days":
            d_max = np.max(d_unique)
            num_mask_d = max(1, int(d_unique.size * 0.15))
            mask_end = d_max
            mask_start = d_max - num_mask_d + 1
            mask |= (d >= mask_start) & (d <= mask_end)
            
        else:
            mask_start = np.random.choice(d_unique)
            mask_end = mask_start + 1
            mask |= (d >= mask_start) & (d <= mask_end)

        mask &= (d != DAY_SEP_TOKEN) # never mask SEP rows
        return mask
        
    def __len__(self):
        return len(self.d_array)

    def get_loc_profile(self, input_x, input_y, city):
        loc_profile = []
        for i in range(input_x.shape[0]):
            if city == 5:
                loc_profile.append(np.zeros(128))
            else:
                loc_profile.append(LOC_PROFILES[city][(input_x[i], input_y[i])])
                
        loc_profile = torch.tensor(np.stack(loc_profile, axis=0), dtype=torch.float32)
        return loc_profile

    def __getitem__(self, index):
        traj_len = int(self.d_array[index].shape[0])
        s, e = self.get_sub_traj(traj_len)

        # slice
        d = self.d_array[index][s:e].copy()
        t = self.t_array[index][s:e].copy()
        dow = self.dow_array[index][s:e].copy()
        time_delta = self.time_delta_array[index][s:e].copy()
        city = self.city_array[index][s:e].copy()
        label_x = self.label_x_array[index][s:e].copy()
        label_y = self.label_y_array[index][s:e].copy()
        trip_id = self.trip_id_array[index][s:e].copy()
        trip_seq = self.trip_seq_array[index][s:e].copy()
        input_x = self.input_x_array[index][s:e].copy()
        input_y = self.input_y_array[index][s:e].copy()
        loc_profile = self.get_loc_profile(input_x, input_y, city[0])
        
        # mask
        mask = self.get_mask(d, t, dow, trip_id)
        input_x[mask] = MASK_TOKEN
        input_y[mask] = MASK_TOKEN
        trip_seq[mask] = PAD_TOKEN
        loc_profile[mask, :] = PAD_TOKEN
        
        return {
            'd': torch.tensor(d, dtype=torch.long),
            't': torch.tensor(t, dtype=torch.long),
            'dow': torch.tensor(dow, dtype=torch.long),
            'input_x': torch.tensor(input_x, dtype=torch.long),
            'input_y': torch.tensor(input_y, dtype=torch.long),
            'time_delta': torch.tensor(time_delta, dtype=torch.long),
            'city': torch.tensor(city, dtype=torch.long),
            'label_x': torch.tensor(label_x, dtype=torch.long),
            'label_y': torch.tensor(label_y, dtype=torch.long),
            'trip_seq': torch.tensor(trip_seq, dtype=torch.long),
            'loc_profile': loc_profile
        }

######################################################
######################################################
######################################################

class FinetuneDataset(BaseDataset):
    def __init__(self, cities, config, is_train=True):
        super().__init__(cities, config)
        self.is_train = is_train
        self.num_train_per_batch = config.TRAIN.NUM_TRAIN_PER_BATCH
        self.mask_type = config.DATA.MASK_TYPE
        self.max_seq_len = config.MODEL.MOBERT.ENCODER.MAX_SEQ_LEN
        self.min_seq_len = 32
        self.tar_cluster = config.DATA.TAR_CLUSTER
        
        for city_name in self.cities:
            print(f"=> Loading data in {city_name}")
            traj_df = load_traj_data(CITY2PATH[city_name])
            user_ids = pd.unique(traj_df['uid'])
            num_users = user_ids.shape[0]
            num_val, num_test = 3000, 3000
            train_user_ids = user_ids[:-(num_test + num_val)]
            val_user_ids = user_ids[-(num_test + num_val): -num_test]
            if self.is_train:
                mask = (traj_df['uid'].isin(val_user_ids)) & (traj_df['d'] > 60)
                traj_df.loc[mask, ['x', 'y']] = 999
                traj_df = traj_df[(traj_df['x'] < 999) & (traj_df['y'] < 999)]
            else:
                val_user_ids = val_user_ids[-config.TRAIN.NUM_VAL:]
                traj_df = traj_df[traj_df['uid'].isin(val_user_ids)]

            for uid, traj in tqdm(traj_df.groupby('uid')):
                if self.tar_cluster >= 0 and USER_CLUSTERS[city_name].get(uid) != self.tar_cluster:
                    continue

                traj_out = process_single_trajectory(city_name, traj)
                self.uid_array.append(uid)
                self.d_array.append(traj_out['d'])  # 1-75; 0:<pad>
                self.t_array.append(traj_out['t'])  # 1-48; 0:<pad>
                self.dow_array.append(traj_out['dow']) # 1-7; 0:<pad>
                self.city_array.append(traj_out['city'])   # add: city ABCD encode 1234
                self.input_x_array.append(traj_out['input_x'])  # 1-200; 0:<pad>; 201:<mask>
                self.input_y_array.append(traj_out['input_y'])  # 1-200; 0:<pad>; 201:<mask>
                self.time_delta_array.append(traj_out['time_delta']) # 1-48; 0:<pad>
                self.label_x_array.append(traj_out['label_x'])  # 0-199
                self.label_y_array.append(traj_out['label_y'])  # 0-199
                self.trip_id_array.append(traj_out['trip_id'])
                self.trip_seq_array.append(traj_out['trip_seq'])

            print(f"City {city_name} - #user = {len(self.uid_array)} in cluster {self.tar_cluster}")
            
    def __len__(self):
        if self.is_train and self.num_train_per_batch > 0:
            return self.num_train_per_batch
        else:
            return len(self.d_array)

    def get_src_traj(self, d, src_end):
        traj_len = len(d)
        src_e = np.where(d <= src_end)[0][-1] + 1
        if src_e < self.max_seq_len:
            return 0, src_e
        else:
            return src_e - self.max_seq_len, src_e

    def get_loc_profile(self, input_x, input_y, city):
        loc_profile = []
        for i in range(input_x.shape[0]):
            if city == 5:
                loc_profile.append(np.zeros(128))
            else:
                loc_profile.append(LOC_PROFILES[city][(input_x[i], input_y[i])])
                
        loc_profile = torch.tensor(np.stack(loc_profile, axis=0), dtype=torch.float32)
        return loc_profile

    def __getitem__(self, _index):

        if self.is_train and self.num_train_per_batch > 0:
            index = random.randint(0, len(self.d_array) - 1)
        else:
            index = _index

        if self.is_train:
            d = self.d_array[index]
            d_unique = np.unique(d[d != DAY_SEP_TOKEN])
            if random.random() > 0.5 and np.max(d_unique) > 60:
                s1, e1 = self.get_src_traj(d, src_end=60)  # last d=60 (actually the last is 76)
                s2, e2 = np.where((d > 60) & (d != DAY_SEP_TOKEN))[0][0], len(d)
            else:
                s2, e2 = len(d) * 3 // 4, len(d)
                s1, e1 = max(0, s2 - self.max_seq_len + 1), s2
        else:
            d = self.d_array[index]
            s1, e1 = self.get_src_traj(d, src_end=60)
            s2, e2 = np.where((d > 60) & (d != DAY_SEP_TOKEN))[0][0], len(d)
    
        d_src = self.d_array[index][s1:e1].copy()
        t_src = self.t_array[index][s1:e1].copy()
        dow_src = self.dow_array[index][s1:e1].copy()
        time_delta_src = self.time_delta_array[index][s1:e1].copy()
        trip_seq_src = self.trip_seq_array[index][s1:e1].copy()
        input_x_src = self.input_x_array[index][s1:e1].copy()
        input_y_src = self.input_y_array[index][s1:e1].copy()
        loc_profile = self.get_loc_profile(input_x_src, input_y_src, self.city_array[index][0])

        mask = (self.d_array[index][s2:e2] == DAY_SEP_TOKEN)
        keep = ~mask

        d_tgt = self.d_array[index][s2:e2][keep].copy()
        t_tgt = self.t_array[index][s2:e2][keep].copy()
        dow_tgt = self.dow_array[index][s2:e2][keep].copy()
        time_delta_tgt = self.time_delta_array[index][s2:e2][keep].copy()
        trip_seq_tgt = self.trip_seq_array[index][s2:e2][keep].copy()
        x_bos = last_valid_token(input_x_src)
        y_bos = last_valid_token(input_y_src)
        input_x_tgt = self.input_x_array[index][s2:e2][keep].copy()
        input_y_tgt = self.input_y_array[index][s2:e2][keep].copy()
        input_x_tgt = np.concatenate([np.array([x_bos]), input_x_tgt[:-1]])
        input_y_tgt = np.concatenate([np.array([y_bos]), input_y_tgt[:-1]])
        label_x_tgt = self.label_x_array[index][s2:e2][keep].copy()
        label_y_tgt = self.label_y_array[index][s2:e2][keep].copy()
        label_sep_token = (trip_seq_tgt == 1).astype(np.int64) # 1st token in each trip
        label_sep_token[0] = 0
        
        return {
            'uid': self.uid_array[index],
            'd': torch.tensor(d_src, dtype=torch.long),
            't': torch.tensor(t_src, dtype=torch.long),
            'dow': torch.tensor(dow_src, dtype=torch.long),
            'time_delta': torch.tensor(time_delta_src, dtype=torch.long),
            'trip_seq': torch.tensor(trip_seq_src, dtype=torch.long),
            'input_x': torch.tensor(input_x_src, dtype=torch.long),
            'input_y': torch.tensor(input_y_src, dtype=torch.long),
            'd_tgt': torch.tensor(d_tgt, dtype=torch.long),
            't_tgt': torch.tensor(t_tgt, dtype=torch.long),
            'dow_tgt': torch.tensor(dow_tgt, dtype=torch.long),
            'time_delta_tgt': torch.tensor(time_delta_tgt, dtype=torch.long),
            'input_x_tgt': torch.tensor(input_x_tgt, dtype=torch.long),
            'input_y_tgt': torch.tensor(input_y_tgt, dtype=torch.long),
            'label_x_tgt': torch.tensor(label_x_tgt, dtype=torch.long),
            'label_y_tgt': torch.tensor(label_y_tgt, dtype=torch.long),
            'label_sep_token': torch.tensor(label_sep_token, dtype=torch.long),
            'loc_profile': loc_profile,
        }



######################################################
######################################################
######################################################

class TestDataset(BaseDataset):
    def __init__(self, cities, config, is_train=False):
        super().__init__(cities, config)
        self.is_train = is_train
        self.num_train_per_batch = config.TRAIN.NUM_TRAIN_PER_BATCH
        self.mask_type = config.DATA.MASK_TYPE
        self.max_seq_len = config.MODEL.MOBERT.ENCODER.MAX_SEQ_LEN
        self.min_seq_len = 32
        self.tar_cluster = config.DATA.TAR_CLUSTER
        
        for city_name in self.cities:
            print(f"=> Loading data in {city_name}")
            traj_df = load_traj_data(CITY2PATH[city_name])
            user_ids = pd.unique(traj_df['uid'])
            num_users = user_ids.shape[0]
            num_test = 3000
            test_user_ids = user_ids[-num_test:]
            traj_df = traj_df[traj_df['uid'].isin(test_user_ids)]

            for uid, traj in tqdm(traj_df.groupby('uid')):
                if self.tar_cluster >= 0 and USER_CLUSTERS[city_name].get(uid) != self.tar_cluster:
                    continue
                    
                traj_out = process_single_trajectory(city_name, traj)
                self.uid_array.append(uid)
                self.d_array.append(traj_out['d'])  # 1-75; 0:<pad>
                self.t_array.append(traj_out['t'])  # 1-48; 0:<pad>
                self.dow_array.append(traj_out['dow']) # 1-7; 0:<pad>
                self.city_array.append(traj_out['city'])   # add: city ABCD encode 1234
                self.input_x_array.append(traj_out['input_x'])  # 1-200; 0:<pad>; 201:<mask>
                self.input_y_array.append(traj_out['input_y'])  # 1-200; 0:<pad>; 201:<mask>
                self.time_delta_array.append(traj_out['time_delta']) # 1-48; 0:<pad>
                self.label_x_array.append(traj_out['label_x'])  # 0-199
                self.label_y_array.append(traj_out['label_y'])  # 0-199
                self.trip_id_array.append(traj_out['trip_id'])
                self.trip_seq_array.append(traj_out['trip_seq'])

            print(f"City {city_name} - #user = {len(self.uid_array)} in cluster {self.tar_cluster}")
            
    def __len__(self):
        return len(self.d_array)

    def get_src_traj(self, d, src_end):
        traj_len = len(d)
        src_e = np.where(d <= src_end)[0][-1] + 1
        if src_e < self.max_seq_len:
            return 0, src_e
        else:
            return src_e - self.max_seq_len, src_e

    def get_loc_profile(self, input_x, input_y, city):
        loc_profile = []
        for i in range(input_x.shape[0]):
            if city == 5:
                loc_profile.append(np.zeros(128))
            else:
                loc_profile.append(LOC_PROFILES[city][(input_x[i], input_y[i])])
                
        loc_profile = torch.tensor(np.stack(loc_profile, axis=0), dtype=torch.float32)
        return loc_profile

    def __getitem__(self, index):

        d = self.d_array[index]
        s1, e1 = self.get_src_traj(d, src_end=60)
        s2, e2 = np.where((d > 60) & (d != DAY_SEP_TOKEN))[0][0], len(d)
    
        d_src = self.d_array[index][s1:e1].copy()
        t_src = self.t_array[index][s1:e1].copy()
        dow_src = self.dow_array[index][s1:e1].copy()
        time_delta_src = self.time_delta_array[index][s1:e1].copy()
        trip_seq_src = self.trip_seq_array[index][s1:e1].copy()
        input_x_src = self.input_x_array[index][s1:e1].copy()
        input_y_src = self.input_y_array[index][s1:e1].copy()
        loc_profile = self.get_loc_profile(input_x_src, input_y_src, self.city_array[index][0])

        sep_mask = (self.d_array[index][s2:e2] == DAY_SEP_TOKEN)
        keep = ~sep_mask

        d_tgt = self.d_array[index][s2:e2][keep].copy()
        t_tgt = self.t_array[index][s2:e2][keep].copy()
        dow_tgt = self.dow_array[index][s2:e2][keep].copy()
        time_delta_tgt = self.time_delta_array[index][s2:e2][keep].copy()
        trip_seq_tgt = self.trip_seq_array[index][s2:e2][keep].copy()

        x_bos = last_valid_token(input_x_src)
        y_bos = last_valid_token(input_y_src)
        input_x_tgt = self.input_x_array[index][s2:e2][keep].copy()
        input_y_tgt = self.input_y_array[index][s2:e2][keep].copy()
        input_x_tgt = np.concatenate([np.array([x_bos]), input_x_tgt[:-1]])
        input_y_tgt = np.concatenate([np.array([y_bos]), input_y_tgt[:-1]])
        label_x_tgt = self.label_x_array[index][s2:e2][keep].copy()
        label_y_tgt = self.label_y_array[index][s2:e2][keep].copy()
        label_sep_token = (trip_seq_tgt == 1).astype(np.int64) # 1st token in each trip
        label_sep_token[0] = 0
        
        return {
            'uid': self.uid_array[index],
            'd': torch.tensor(d_src, dtype=torch.long),
            't': torch.tensor(t_src, dtype=torch.long),
            'dow': torch.tensor(dow_src, dtype=torch.long),
            'time_delta': torch.tensor(time_delta_src, dtype=torch.long),
            'trip_seq': torch.tensor(trip_seq_src, dtype=torch.long),
            'input_x': torch.tensor(input_x_src, dtype=torch.long),
            'input_y': torch.tensor(input_y_src, dtype=torch.long),
            'd_tgt': torch.tensor(d_tgt, dtype=torch.long),
            't_tgt': torch.tensor(t_tgt, dtype=torch.long),
            'dow_tgt': torch.tensor(dow_tgt, dtype=torch.long),
            'time_delta_tgt': torch.tensor(time_delta_tgt, dtype=torch.long),
            'input_x_tgt': torch.tensor(input_x_tgt, dtype=torch.long),
            'input_y_tgt': torch.tensor(input_y_tgt, dtype=torch.long),
            'label_x_tgt': torch.tensor(label_x_tgt, dtype=torch.long),
            'label_y_tgt': torch.tensor(label_y_tgt, dtype=torch.long),
            'label_sep_token': torch.tensor(label_sep_token, dtype=torch.long),
            'loc_profile': loc_profile,
        }
