#!/bin/bash
source ~/.bashrc
conda create -y --name Milbinding
conda activate Milbinding
conda install -y python=3.10.9
conda install -y numpy=1.22.3
pip install scikit-learn
conda install -y matplotlib=3.7.1
conda install -y seaborn
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
conda install -y pyg -c pyg
pip install e3nn
pip install torch-cluster 
pip install torch-scatter 
pip install mdtraj
