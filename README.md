# Mode recovery in neural autoregressive sequence modeling

## ACL-IJCNLP 2021 5th Workshop on Structured Prediction for NLP

Paper link: https://arxiv.org/abs/2106.05459


## Usage

This code is the implementation of the learning chain used in the paper. Single experiment with a particular learning chain can be started as:

`python run_experiment.py N`,

where N is the number starting from 1 to 300 meaning the particular configuration of the learning chain returned by the `config_factory()` function.

Every experiment saves the statistics from all induced distribution along the learning chain in the pickle file. Mode recovery costs are computed using functions from `compute_pkls.py`. `compute_pkls.py` parallelize computation of mode recovery costs across multiple cpus such that each process takes an independent pickle file.

The jupyter notebook `plots.ipynb` implements all the plots which are used in the paper. Please contact me if you want to get pkl files from our experiments.

## Requirements

* torch
* scipy
* numpy
* tqdm
* fire
