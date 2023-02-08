#!/bin/bash

for seed in 1 6 17 38 99
do
    python main.py --cfg configs/StructuralAwareness/triangle/triangle-gin-t1.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/triangle/triangle-gin-t2.yaml --repeat 1 seed $seed

    python main.py --cfg configs/StructuralAwareness/triangle/triangle-tf-t1.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/triangle/triangle-tf-t2.yaml --repeat 1 seed $seed

    python main.py --cfg configs/StructuralAwareness/triangle/triangle-tf-lap-t1.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/triangle/triangle-tf-lap-t2.yaml --repeat 1 seed $seed

    python main.py --cfg configs/StructuralAwareness/triangle/triangle-tf-rwse-t1.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/triangle/triangle-tf-rwse-t2.yaml --repeat 1 seed $seed

    python main.py --cfg configs/StructuralAwareness/triangle/triangle-graphormer-t1.yaml --repeat 1 seed $seed
    python main.py --cfg configs/StructuralAwareness/triangle/triangle-graphormer-t2.yaml --repeat 1 seed $seed
done
