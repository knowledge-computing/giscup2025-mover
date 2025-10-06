import os
import json
from utils import seq_eval

out_generated = []
out_referenced = []

city = "D"
for i in range(4):
    with open(f"_validation/finetune_cityA___city{city}__dual__cluster{i}.json") as f:
        json_data = json.load(f)
        out_generated += json_data['generated']
        out_referenced += json_data['reference']
        unique_user = list(set(gen[0] for gen in json_data['generated']))
        print(len(unique_user))

unique_user = list(set(gen[0] for gen in out_generated))
print(len(unique_user))

result = {'generated': out_generated, 'reference': out_referenced}
geobleu_val = seq_eval.calc_geobleu_bulk(result['generated'], result['reference'])
dtw_val = seq_eval.calc_dtw_bulk(result['generated'], result['reference'])
print("Geobleu: {}".format(geobleu_val))
print("DTW: {}".format(dtw_val))

with open(f"_validation/finetune_cityA___city{city}__dual__cluster.json", 'w') as f:
    json.dump(result ,f)