#!/bin/bash

# Edges
for seed in 1 6 17 38 99
do
    python main.py --cfg configs/StructAware2D/edge-graphormer.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructAware2D/edge-tf.yaml --repeat 1 seed $seed posenc_LapPE.enable True dataset.node_encoder_name TypeDictNode+LapPE
    python main.py --cfg configs/StructAware2D/edge-tf.yaml --repeat 1 seed $seed posenc_RWSE.enable True dataset.node_encoder_name TypeDictNode+RWSE
    python main.py --cfg configs/StructAware2D/edge-tf.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructAware2D/edge-gin.yaml --repeat 1 seed $seed
done

# Triangles
for seed in 1 6 17 38 99
do
    python main.py --cfg configs/StructAware2D/triangle-graphormer.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructAware2D/triangle-tf-lap.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructAware2D/triangle-tf-rwse.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructAware2D/triangle-tf.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructAware2D/triangle-gin.yaml --repeat 1 seed $seed
done

# CSL
for seed in 1 6 17 38 99
do
    python main.py --cfg configs/StructAware2D/csl-graphormer.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructAware2D/csl-tf.yaml --repeat 1 seed $seed posenc_LapPE.enable True dataset.node_encoder_name TypeDictNode+LapPE
    python main.py --cfg configs/StructAware2D/csl-tf.yaml --repeat 1 seed $seed posenc_RWSE.enable True dataset.node_encoder_name TypeDictNode+RWSE
    python main.py --cfg configs/StructAware2D/csl-tf.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructAware2D/csl-gin.yaml --repeat 1 seed $seed
done


