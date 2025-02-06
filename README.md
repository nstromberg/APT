# Adversarially Pre-trained Transformer


## Installation

### 1. Create Conda Environment
```bash
conda create -n apt-env python=3.10
conda activate apt-env
```

### 2. Install Learning Library
- [Pytorch](https://pytorch.org/) [**2.3**.1](https://pytorch.org/get-started/previous-versions/)

  \* *make sure to install the right versions for your toolkit*

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```


## Run
Once the environment is set up, the bash call to pre-train an APT model is:

```bash
./main.sh
```

A list of flags may be found in `main.sh` and `main.py` for experimentation with different hyperparameters. The run log is saved under `logs/`, models are saved under `*artifact_path*/saves`, and the tensorboard log is saved under `*artifact_path*/runs`.
