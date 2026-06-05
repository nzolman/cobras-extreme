# Uncovering Extreme Event Mechanisms for Prediction and Control with Sensitivity-Balanced Projections
![Project Logo](./assets/figure_1.png)

- Preprint: https://arxiv.org/abs/2606.05618


## Abstract
Extreme events---such as earthquakes and coronal mass ejections---are common in many chaotic dynamical systems, yet are difficult to characterize and predict due to the subtle instability mechanisms that drive them.  
In this work, we develop an interpretable technique that reveals the underlying mechanisms behind extreme events and uses them to build data-driven forecasts and intuitive event suppression controllers.
In particular, we utilize the covariance balancing reduction using adjoint snapshots (CoBRAS) method to identify linear oblique projections that best capture the sensitivity of a quantity of interest and reconstruct the original state. 
Importantly, we bypass the need for cumbersome adjoint calculations, instead using backpropagation via modern automatically differentiable numerical frameworks.  
To accommodate spatially localized events, we also introduce a new variant of CoBRAS to obtain local sensitivity-balanced projections. 
We demonstrate the utility of this approach to characterize extreme events across a diverse set of challenging systems, including turbulent bursts of energy dissipation in the 2D Kolmogorov Flow, spontaneous synchronization in networks of coupled FitzHugh-Nagumo oscillators, and
the localized formation of ocean rogue waves from a modified nonlinear Schr{\"o}dinger equation.
For each example, we show that our simple forecast models accurately predict extreme events and that the underlying mechanisms may be used to design control laws to prevent these events.
Finally, we demonstrate that by learning a neural network surrogate model of the dynamics directly from data, we may extend this approach to experimental systems and systems that are not natively written in an automatically differentiable programming language.  

# Code

## Purpose
This repository serves as the home for the code accompanying ``Uncovering Extreme Event Mechanisms for Prediction
and Control with Sensitivity-Balanced Projections'' by Nicholas Zolman, Sajeda Mokbel, Samuel E. Otto, and Steven L. Brunton. 

This repository is not meant to serve as an official standalone package, and there is no expectation it will be officially maintained. However, please feel free to submit github issues and start github discussions! We will attempt to address as much as possible. 

## Installation
The code in this repository has only been verified to run using python 3.12 and (3.12.7 specifically), but is expected to work with newer version. The three simulators we provide are based out of JAX for autodiff compatibility. Kolmogorov flow and MNLS are natively built here, FHN requires [`diffrax`](https://docs.kidger.site/diffrax/). SVMs are built using scikit-learn. We provide a noncomprehensive set of requirements can be found in `requirements.txt`. To use the code as a package, simply run:

```bash
pip install -r requirements.txt
pip install -e .
```

## Tutorial
A complete tutorial for the FitzHugh-Nagumo ($N=101$) example can be found in `tutorials/FHN.ipynb`. This tutorial is completely self-contained; it does not use reference anything else in the package except for plotting defaults, and does not require any external data. It was tested with an NVIDIA GeForce RTX 2080 Ti GPU and ran in just a few minutes with 64-bit precision. Note that we use a smaller dataset than what we used in the paper and that random seeds across different python/pacakge versions and computaitonal hardware can produce slightly different results.

## Data
Accompanying data can be found here: https://huggingface.co/datasets/nzolman/cobras_extreme


## Setting up Environment for The Kolmogorov Flow
To run the Kolmogorov flow examples, you need to include [Controlling-Kolmogorov-Flow](https://github.com/smokbel/Controlling-Kolmogorov-Flow) as a directory in the root of this folder:

```
Controlling-Kolmogorov-Flow/
src/
.gitignore
pyproject.toml
README.md
requirements.text
```

This folder is added to the path in `src/cobras_extreme/__init__.py` so any confusing imports such as

```python
import equations.base as base
from solvers import transient
```

are coming from files in `Controlling-Kolmogorov-Flow/`. 

## Running the Fourier Neural Operator
The code for the FNO can be found in `src/cobras_extreme/kflow/fno` with its own set of dependencies. Weights for the operator found in the paper can be found in the HF dataset. 

# Citing
Please consider citing our paper if you use this work: 

```bibtex
@misc{zolman2026uncovering,
      title={Uncovering Extreme Event Mechanisms for Prediction and Control with Sensitivity-Balanced Projections}, 
      author={Nicholas Zolman and Sajeda Mokbel and Samuel E. Otto and Steven L. Brunton},
      year={2026},
      eprint={2606.05618},
      archivePrefix={arXiv},
      primaryClass={nlin.CD},
      url={https://arxiv.org/abs/2606.05618}, 
}
```