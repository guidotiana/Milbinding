# Welcome to Milbinding

Welcome to Milbinding. This repository aims to provide scripts for reproducing the results from _"Predicting the binding of small molecules to proteins through invariant representation of the molecular structure"_ - R. Beccaria et al.

![alt text](https://github.com/guidotiana/Milbinding/blob/main/pic.png?raw=true)

# Installation

**1. Python Libraries**
   
We provide a script for installing all the (python) libraries required to use our equivariant graph autoencoder via Anaconda. After activating your base environment, type in your terminal:

`$ ./install_packages_OSX.sh` if your operating system is macOS

`$ ./install_packages_LINUX.sh` if your operating system is Linux

The script will create a conda environment named *Milbinding* which contains all the dependencies to reproduce our results and to use our equivariant graph autoencoder. You can activate the environment by typing:

`$ conda activate Milbinding`


**2. Datasets for reproducing results**

Once the Milbinding conda environment is installed, the datasets and the trained models can be downloaded. Type in your terminal:

`$ download_datas.sh`

This bash script will automatically download all the datasets and trained models from the UNIMI Datavers. All the datasets and trained models are saved into the folder _datasets_and_RFs_ 

**3. Reproduce results**

With the Milbinding conda environment activated, the results published in the paper can be reproduced by running in the terminal:

`$ python Reproduce_Results.py`

This Python script will display in the terminal our main results and will save in the folder _images_results_ all the figures reported in the paper with their relative results.

N.B. It can happen that by running the _Reproduce_Results.py_ python script, the execution fails due to some files not being found in the _datasets_and_RFs_ folder. This could be related to some failures in downloading all the files from the UNIMI Datavers using the provided bash script _download_datas.sh_. To fix this, we recommend visiting the following website (https://dataverse.unimi.it/dataset.xhtml?persistentId=doi:10.13130/RD_UNIMI/5879ZG), manually downloading the missing files and moving them into the _datasets_and_RFs_ folder.

# Tutorial

In this Repository, we also provide a Tutorial on how to use our trained equivariant graph encoder to obtain the encoded matrixes (fingerprints) discussed in the paper.
We also show how to use the trained Random Forest to make a binding prediction between a ligand and a pocket using the encoded matrixes. The tutorial might be found in the _Milbinding.ipynb_ notebook
