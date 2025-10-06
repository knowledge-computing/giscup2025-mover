import os

# CITY2PATH = {
#     'cityA': '/home/yaoyi/shared/humob/giscup2025/data/processed_data/city_A_challengedata.csv.gz',
#     'cityB': '/home/yaoyi/shared/humob/giscup2025/data/processed_data/city_B_challengedata.csv.gz',
#     'cityC': '/home/yaoyi/shared/humob/giscup2025/data/processed_data/city_C_challengedata.csv.gz',
#     'cityD': '/home/yaoyi/shared/humob/giscup2025/data/processed_data/city_D_challengedata.csv.gz'
# }

MASK_TOKEN = 201
PAD_TOKEN = 0
DAY_SEP_TOKEN = 76
TIME_SEP_TOKEN = 49
DOW_SEP_TOKEN = 8
XY_SEP_TOKEN = 202
CITY_SEP_TOKEN = 5

root = "/projects/standard/yaoyi/shared/humob/giscup2025/trip_detection/trip_detection_v2_append_processed/"
CITY2PATH = {
    'cityA': os.path.join(root, 'trajectories_with_trips_cityA.csv'),
    'cityB': os.path.join(root, 'trajectories_with_trips_cityB.csv'),
    'cityC': os.path.join(root, 'trajectories_with_trips_cityC.csv'),
    'cityD': os.path.join(root, 'trajectories_with_trips_cityD.csv'),
}

CITY2ID = {
    'cityA': 1,
    'cityB': 2,
    'cityC': 3,
    'cityD': 4
}
