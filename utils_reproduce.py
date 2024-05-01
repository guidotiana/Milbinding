import numpy as np
#import os
from tqdm import tqdm
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn import svm
import pickle as cPickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import seaborn as sns
from scipy import optimize
import scipy as sp


import math
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader

from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
from Autoencoder import Autoencoder_cmap
import torch
import mdtraj as md

def fit_function(x, a, b):
    return a + b * x

def fit_function1(x,a,b):
    return np.exp(x/a)+b

def fit_function2(x,a,b,c):
    return c*np.exp(x/a)+b

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer, output_dim, activ_func, dropoutprob):
        super(FeedforwardNeuralNetModel, self).__init__()
        modules = []
        modules.append(torch.nn.Flatten())
        if activ_func=="ELU":
            modules.append(torch.nn.Linear(input_dim, hidden_dim))
            modules.append(torch.nn.ELU())
        if activ_func=="Tanh":
            modules.append(torch.nn.Linear(input_dim, hidden_dim))
            modules.append(torch.nn.Tanh())
        if activ_func=="Linear":
            modules.append(torch.nn.Linear(input_dim, hidden_dim))
        modules.append(torch.nn.Dropout(p=dropoutprob))
        for i in range(hidden_layer-1):
            if activ_func=="ELU":
                modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
                modules.append(torch.nn.ELU())
            if activ_func=="Tanh":
                modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
                modules.append(torch.nn.Tanh())
            if activ_func=="Linear":
                modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
            modules.append(torch.nn.Dropout(p=dropoutprob))
        modules.append(torch.nn.Linear(hidden_dim,output_dim))
        

        self.NN = torch.nn.Sequential(*modules)
        
    def forward(self, item):
        return(self.NN(item))
    
    

def KL_symm(matrix1, matrix2):
    KL1=0
    KL2=0
    num1=matrix1.sum()
    num2=matrix2.sum()
    L=len(matrix1)
    for i in range(len(matrix1)):
        KL1+=(matrix1[i]+1)*(np.log((matrix1[i]+1)*(num2+L)/((num1+L)*(matrix2[i]+1))))/(num1+L)
    for i in range(len(matrix2)):
        KL2+=(matrix2[i]+1)*(np.log((matrix2[i]+1)*(num1+L)/((num2+L)*(matrix1[i]+1))))/(num2+L)
    return 0.5*(KL1+KL2)



def get_ligand_from_name(name_now):

    device="cpu"
    net_ligand = SimpleNetwork(
        irreps_in="9x0e",
        irreps_out="1x0e",
        max_radius=1.0,
        num_neighbors=10,
        num_nodes=30,
        layers=3, #Da matchare
        lmax=3, #Da matchare
        mul=8, #Da matchare
        pool_nodes=False
    )
    min_val_filename_ligand="batch_size=128_activation_func_mlp_cmap=ELU_activation_func_mlp_atoms=ELU_n_nodes_mlp_cmap=500_n_nodes_mlp_atoms=1_n_layers_mlp_cmap=3_n_layers_mlp_atoms=1_max_radius_e3nn=1.0_n_layers_e3nn=3_lmax_e3nn=3_mul_e3nn=8_wd=0.0_n_epochs=200_constant=1.0"
    net_r_ligand = Autoencoder_cmap(net_ligand, 30, 9, 128, "ELU", 500, 3, device)
    net_r_ligand.load_state_dict(torch.load("./equivariant_autoencoder/"+min_val_filename_ligand+"/model_val.pt", map_location=torch.device(device)))

    Name_list=np.load("./datasets_and_RFs/name_ligand.npy")
    Bins_list=np.load("./datasets_and_RFs/bins_ligand.npy")
    num_bins=40

    
    pdb = md.load_pdb("./datasets_and_RFs/"+name_now+".pdb")
    data=pdb.xyz
    atom_type=[]
    for atom in pdb.topology.atoms:
        if atom.name[0:1]=="C" or atom.name[0:1]=="c":
            if atom.name[1:2]=="L" or atom.name[1:2]=="l":
                atom_type.append("CL")
            else:
                atom_type.append(atom.name[0:1])
        else:
            atom_type.append(atom.name[0:1])
        
    my_set_for_filter_ligand=np.load("./datasets_and_RFs/filters_names_bins_my_set_for_filter_lig.npy")
    my_filter_ligand=np.load("./datasets_and_RFs/filters_names_bins_my_filter_lig.npy", allow_pickle=True)
    pos = torch.from_numpy(data)
    pos = pos.to(torch.float32)
    x = torch.zeros(len(atom_type), len(my_set_for_filter_ligand))
    atom_names_ligand=[]
    for j,name in enumerate(atom_type):
        atom_names_ligand.append(name)
        x[j]=my_filter_ligand.item()[name]
    pos=pos.squeeze()
    pos = pos.to(device)
                   
    lat_vec=net_r_ligand.SN_cmap({"x":x, "pos":pos}).detach().cpu().numpy()
    
    Matrix_lig=np.zeros((len(Name_list),num_bins+2))
    for j,atom in enumerate(lat_vec):
        for w in range(len(Name_list)):
            if atom_names_ligand[j]==Name_list[w]:
                index=int(np.digitize(lat_vec[j], Bins_list[w]))
                Matrix_lig[w,index]+=1

    return Matrix_lig

