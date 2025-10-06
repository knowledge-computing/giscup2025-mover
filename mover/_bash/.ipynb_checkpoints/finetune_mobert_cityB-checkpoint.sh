for cluster in 0 1 2 3 4; do
    echo "Running for Cluster=$cluster"

    python finetune.py \
        --config configs/finetune/finetune_cityA___cityB_cluster.yaml \
        --exp_name finetune_cityA___cityB__dual__cluster${cluster} \
        --use_wandb \
        --tar_cluster $cluster

    python validate.py \
        --config _runs/finetune_cityA___cityB__dual__cluster${cluster}/config.yaml
done

