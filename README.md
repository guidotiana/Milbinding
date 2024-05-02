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


