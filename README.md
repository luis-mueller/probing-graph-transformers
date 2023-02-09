# Attending to Graph Transformers

[![arXiv](https://img.shields.io/badge/arXiv-2302.04181-b31b1b.svg)](https://arxiv.org/abs/2302.04181)

Code for our paper [Attending to Graph Transformers](https://arxiv.org/abs/2302.04181). We base our implementation on the [GraphGPS](https://github.com/rampasek/GraphGPS) repository.
GraphGPS is built using [PyG](https://www.pyg.org/) and [GraphGym from PyG2](https://pytorch-geometric.readthedocs.io/en/2.0.0/notes/graphgym.html). Specifically *PyG v2.2* is required.

The paper presents three different experiments, probing...

- the structural awareness of different structural biases (positional/structural encodings, attention bias) to properties of the graph, such as adjacency, number of triangles, etc.
- their ability to prevent over-smoothing on heterophilic datasets Actor, Cornell, Texas, Wisconsin, Chameleon and Squirrel.
- their ability to prevent over-squashing on the NeighborsMatch problem of [Alon and Yahav, 2021](https://arxiv.org/abs/2006.05205).


### Python environment setup with Conda

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```

### Running an experiment with GraphGPS
```bash
conda activate graphgps

# Running an arbitrary config file in the `configs` folder
python main.py --cfg configs/GPS/<config_file>.yaml  wandb.use False
```
We provide the config files necessary to reproduce our experiments under `configs/` (see more below).

### W&B logging
To use W&B logging, set `wandb.use True` and have a `gtransformers` entity set-up in your W&B account (or change it to whatever else you like by setting `wandb.entity`).

### Structural Awareness of GTs
We prepared config files to reproduce the structural awareness experiments under `configs/StructuralAwareness`.
The experimets are performed on three tasks, `Edges`, `Triangles`, `CSL`. In addition, the test set of the `Triangles` task contains both small and large graphs and we benchmark performance for them separately, resulting in `Triangles-small` and `Triangles-large` in the paper. The precise commands used to run these experiments can be found in `run/run_structure_awareness.sh`. To benchmark the `Triangles-small` and `Triangles-large` separately, first run `run/run_structure_awareness.sh` and then copy the folder generated for the `Triangles` runs under `results` into a new folder called `pretrained` and run `run/run_triangles_small_large_split.sh`.

### Reduced Over-smoothing in GTs?
Similar to the structural awareness experiments, we prepared config files to reproduce the experiments on heterophilic datasets under `configs/GPS` and `configs/Graphormer` for Transformer with positional/structural encodigns and optional message-passing and Graphormer, respectively. The precise commands used to run our experiments, including the commands for our hyper-parameter search, can be found in `run/run_heterophilic.sh`.

### Reduced Over-squashing in GTs?
To reproduce our results on the `NeighborsMatch` dataset, visit our dedicated fork at [https://github.com/luis-mueller/bottleneck](https://github.com/luis-mueller/bottleneck), which we set up to stay as close as possible to the original implementation in [Alon and Yahav, 2021](https://arxiv.org/abs/2006.05205).


## Unit tests

To run all unit tests, execute from the project root directory:

```bash
python -m unittest -v
```

Or specify a particular test module, e.g.:

```bash
python -m unittest -v unittests.test_eigvecs
```

## Citation
If you find this work useful, please cite 

```bibtex
@article{mueller2023attending,
  title={{Attending to Graph Transformers}}, 
  author={Luis Müller and Christopher Morris and Mikhail Galkin and Ladislav Ramp\'{a}\v{s}ek},
  journal={Arxiv preprint},
  year={2023}
}
```

and the GraphGPS paper:
```bibtex
@article{rampasek2022GPS,
  title={{Recipe for a General, Powerful, Scalable Graph Transformer}}, 
  author={Ladislav Ramp\'{a}\v{s}ek and Mikhail Galkin and Vijay Prakash Dwivedi and Anh Tuan Luu and Guy Wolf and Dominique Beaini},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```
