#!/bin/bash -l 
#SBATCH --time=23:00:00 
#SBATCH -N 1
#SBATCH --ntasks-per-node=8 
#SBATCH --mem=60GB 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=lin00786@umn.edu 
#SBATCH -p yaoyi 
#SBATCH --gres=gpu:a100:1 


module load python3 
module load gcc/13.1.0-mptekim 
source activate map 

cd /home/yaoyi/lin00786/work/Humob/mobert

# python finetune.py --config configs/finetune/finetune_cityA.yaml --use_wandb

# for cluster in 0 1 2 3 4; do
#     echo "Running for Cluster=$cluster"

#     python finetune.py \
#         --config configs/finetune/finetune_cityA___cityA_cluster.yaml \
#         --exp_name finetune_cityA___cityA__dual__cluster${cluster} \
#         --use_wandb \
#         --tar_cluster $cluster

#     python validate.py \
#         --config _runs/finetune_cityA___cityA__dual__cluster${cluster}/config.yaml
# done


for cluster in 0 1 2 3 4; do
    echo "Running for Cluster=$cluster"

    python finetune.py \
        --config configs/finetune/finetune_cityA___cityA_cluster.yaml \
        --exp_name finetune_cityA___cityA__dual__cluster${cluster}_v2 \
        --use_wandb \
        --tar_cluster $cluster \
        --pretrain_weight /home/yaoyi/lin00786/work/Humob/mobert/_runs/finetune_mobert___seq768__loc_profile__cityABCD__dual_conditional__cityA_v2/checkpoint/model_best.pth

    python validate.py \
        --config _runs/finetune_cityA___cityA__dual__cluster${cluster}_v2/config.yaml
done