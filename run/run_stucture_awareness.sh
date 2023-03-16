#!/bin/bash

# Edges
for seed in 1 6 17 38 99
do
    python main.py --cfg configs/StructuralAwareness/edge-graphormer.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/edge-tf.yaml --repeat 1 seed $seed posenc_LapPE.enable True dataset.node_encoder_name TypeDictNode+LapPE
    python main.py --cfg configs/StructuralAwareness/edge-tf.yaml --repeat 1 seed $seed posenc_RWSE.enable True dataset.node_encoder_name TypeDictNode+RWSE
    python main.py --cfg configs/StructuralAwareness/edge-tf.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/edge-gin.yaml --repeat 1 seed $seed
done

# Triangles
for seed in 1 6 17 38 99
do
    python main.py --cfg configs/StructuralAwareness/triangle-graphormer.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/triangle-tf-lap.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/triangle-tf-rwse.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/triangle-tf.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/triangle-gin.yaml --repeat 1 seed $seed
done

# CSL
for seed in 1 6 17 38 99
do
    python main.py --cfg configs/StructuralAwareness/csl-graphormer.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/csl-tf.yaml --repeat 1 seed $seed posenc_LapPE.enable True dataset.node_encoder_name LapPE posenc_LapPE.dim_pe 64
    python main.py --cfg configs/StructuralAwareness/csl-tf.yaml --repeat 1 seed $seed posenc_RWSE.enable True dataset.node_encoder_name RWSE posenc_RWSE.dim_pe 64
    python main.py --cfg configs/StructuralAwareness/csl-tf.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/csl-gin.yaml --repeat 1 seed $seed
done


