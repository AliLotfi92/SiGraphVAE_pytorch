# gae-pytorch
Graph Auto-Encoder in PyTorch

This is a PyTorch implementation of the Variational Graph Auto-Encoder model described in the paper:
 
T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)

The code in this repo is based on or refers to https://github.com/tkipf/gae, https://github.com/tkipf/pygcn and https://github.com/vmasrani/gae_in_pytorch.

### Requirements
scipy==1.0.0
numpy==1.14.0
torch==1.4.0
networkx==2.1
scikit_learn==0.19.2
tensorboard==2.2.1
```
pip install -r requirements.txt``` 

### How to run
```bash
python gae/train.py
```
