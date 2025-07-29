<div align="center">

# Enhancing Generalization in Continuous‑Time Flow‑Matching via PAC‑Bayesian Information Bottleneck

[![OT‑CFM Preprint](http://img.shields.io/badge/paper-arxiv.2302.00482-B31B1B.svg)](https://arxiv.org/abs/2302.00482)

</div>

## Overview

**TorchCFM** implements Conditional Flow Matching (CFM), a simulation‑free training objective for continuous normalizing flows (CNFs).  
By integrating a PAC‑Bayesian Information Bottleneck, TorchCFM bounds the generalization gap, controls velocity‑field complexity, and enhances sample robustness.

## Installation

```bash
# Clone repository
git clone https://github.com/atong01/conditional-flow-matching.git
cd conditional-flow-matching

# (Optional) Create and activate conda environment
conda create -n torchcfm python=3.10
conda activate torchcfm

# Install dependencies
pip install -r requirements.txt

# Install torchcfm as a package
pip install -e .
```

## Running Experiments

All training scripts use Hydra for configuration. You can override any parameter via `+key=value`.

```bash
# Unconditional ImageNet64
python scripts/train_imagenet.py

# Conditional ImageNet64
python scripts/train_cond_imagenet.py

# Unconditional CIFAR‑10
python scripts/train_cifar10.py

# Conditional CIFAR‑10
python scripts/train_cond_cifar10.py

# Conditional MNIST
python scripts/train_cond_mnist.py
```

## Key Configuration Parameters

When running any of the training scripts, you can control TorchCFM’s behavior via command‑line flags:

- **`--matcher`**  
  Choose the flow‑matching variant:  
  - `cfm` — ConditionalFlowMatcher  
  - `ot` — ExactOptimalTransportConditionalFlowMatcher  
  - `sb` — SchrödingerBridgeConditionalFlowMatcher  
  - `target` — TargetConditionalFlowMatcher  

- **`--use_ib`**  
  Enable the PAC‑Bayesian Information Bottleneck regularizer.  

- **`--ib_lambda`** (default: `5e-2`)  
  Weight of the kinetic (energy) penalty in the IB‑Flow loss. Larger values enforce lower‑norm velocity fields.  

- **`--ib_beta`** (default: `2e-5`)  
  Weight of the entropy bonus in the IB‑Flow loss. Larger values encourage higher entropy (more diverse dynamics).  

- **`--lr`** (default: `5e-5`)  
  Base learning rate for the optimizer.  

- **`--batch_size`** (default: `64`)  
  Number of samples per training minibatch.  

- **`--total_steps`** (default: `400000`)  
  Total number of training iterations.  

- **`--n_steps`** (default: `128`)  
  Number of ODE solver evaluations when sampling from the learned flow.  

