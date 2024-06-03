# Nonsmooth Implicit Differentiation

Code for the paper 
[Nonsmooth Implicit Differentiation: Deterministic and Stochastic Convergence Rates](https://arxiv.org/abs/2403.11687) by Riccardo Grazzi, Massimiliano Pontil and Saverio Salzo (ICML 2024).


## Getting started
**Install** the packages in [requirements.txt](requirements.txt). We suggest to use a GPU to speed up the computation for data poisoning. 

Check out [elastic_net_toy.ipynb](elastic_net_toy.ipynb) for a illustrative comparison between deterministic AID and ITD derivative approximation methods for (nonsmooth) elastic net.

## How to reproduce results
**Run** one of the following files:
   - [elastic_net_deterministic.py](elastic_net_deterministic.py) for the experiments comparing AID and ITD on computing the derivative with respect to the hyperparameters of elastic net.
   - [elastic_net_stochastic.py](elastic_net_stochastic.py) for the experiments comparing AID-FP and NSID and SID on computing the derivative with respect to the hyperparameters of elastic net.   
   - [data_poisoning_stochastic.py](data_poisoning_stochastic.py) for the experiments comparing (N)SID with constant and decreasing step-size schedules on computing the derivative with respect to the corruption noise of the data poisoning with elastic net regularization.  

Experiments will be saved in the exps folder inside the project directory, while for data poisoning, MNIST will be downloaded in the data folder.

## How to analyse results
Use the notebooks in the [analyse_results](analyse_results) folder to generate plots from the data of previously run experiments.


## Additional info
See [hypertorch](https://github.com/prolearner/hypertorch) for more details on the AID and ITD hypergradient approximation methdos and some examples on how to incorporate them in a project.

[nonsmooth_implicit_diff/stoch_hg.py](nonsmooth_implicit_diff/stoch_hg.py) contains the code for the NSID method to approximate the derivaive of a fixed point which is a composition of an outer map and an inner map accessible only through a stochastic unbiased estimator.

Details on the experimental settings can be found in [our paper](https://arxiv.org/abs/2403.11687). 

## Cite us
If you use this code, please cite [our paper](https://arxiv.org/abs/2403.11687).

```
@article{grazzi2024nonsmooth,
  title={Nonsmooth implicit differentiation: Deterministic and stochastic convergence rates},
  author={Grazzi, Riccardo and Pontil, Massimiliano and Salzo, Saverio},
  journal={arXiv preprint arXiv:2403.11687},
  year={2024}
}