#!/bin/bash
source ~/.bashrc
conda create -y --name Milbinding
source activate Milbinding
conda install -y python=3.10.9
conda install -y numpy=1.22.3
# scikit-learn version 1.2.2
pip install scikit-learn
conda install -y matplotlib=3.7.1
conda install -y seaborn
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
conda install -y pyg -c pyg
pip install e3nn
pip install torch-cluster 
pip install torch-scatter 
pip install mdtraj
pip install notebook
