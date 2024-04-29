from Autoencoder import Autoencoder_cmap
from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
import torch
import mdtraj as md
import numpy as np


def Load_graph_auto_pocket(device="cpu"):

    net_pocket = SimpleNetwork(
        irreps_in="6x0e",
        irreps_out="1x0e",
        max_radius=1.0,
        num_neighbors=15,
        num_nodes=30,
        layers=3, #Da matchare
        lmax=3, #Da matchare
        mul=8, #Da matchare
        pool_nodes=False
    )
    min_val_filename="batch_size=8_activation_func_mlp_cmap=ELU_activation_func_mlp_atoms=ELU_n_nodes_mlp_cmap=500_n_nodes_mlp_atoms=1_n_layers_mlp_cmap=3_n_layers_mlp_atoms=1_max_radius_e3nn=1.0_n_layers_e3nn=3_lmax_e3nn=3_mul_e3nn=8_wd=0.0_n_epochs=500_constant=1.0"
    net_r_pocket = Autoencoder_cmap(net_pocket, 30, 6, 8, "ELU", 500, 3, device) #Da matchare batchsize(1uarto num), 30=max_n_atoms, 4=num_atoms_type, 50=num_nodes, 2=num_layers
    net_r_pocket.load_state_dict(torch.load("./equivariant_autoencoder/"+min_val_filename+"/model_val.pt", map_location=torch.device(device)))
    
    return net_r_pocket
    
    
def Load_graph_auto_ligand(device="cpu"):

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
    
    return net_r_ligand
    
    
    
    
def Get_latent_ligand(pdb_ligand, net_r_ligand, device="cpu"):

    my_set_for_filter_ligand=np.load("../train_equivariant_cloro/all_datasets/filters_names_bins/my_set_for_filter_lig.npy")
    my_filter_ligand=np.load("../train_equivariant_cloro/all_datasets/filters_names_bins/my_filter_lig.npy", allow_pickle=True)
    
    pdb = md.load_pdb(pdb_ligand)
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

    return lat_vec, atom_names_ligand
    
    
def Get_latent_pocket(pdb_pocket, net_r_pocket, device="cpu"):

    my_set_for_filter_pocket=np.load("../train_equivariant_cloro/all_datasets/filters_names_bins/my_set_for_filter_pock.npy")
    my_filter_pocket=np.load("../train_equivariant_cloro/all_datasets/filters_names_bins/my_filter_pock.npy", allow_pickle=True)
    
    pdb = md.load_pdb(pdb_pocket)
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
        
    
    pos = torch.from_numpy(data)
    pos = pos.to(torch.float32)
    x = torch.zeros(len(atom_type), len(my_set_for_filter_pocket))
    atom_names_pocket=[]
    for j,name in enumerate(atom_type):
        atom_names_pocket.append(name)
        x[j]=my_filter_pocket.item()[name]
    pos=pos.squeeze()
    pos = pos.to(device)
                   
    lat_vec=net_r_pocket.SN_cmap({"x":x, "pos":pos}).detach().cpu().numpy()

    return lat_vec, atom_names_pocket


def Get_encoded_matrix_ligand(latent_vec_ligand, atoms_ligand, num_bins=40):

    #Name_list=np.load("../dataset_latent_space_MUV/name_pocket.npy")
    Name_list_ligand=np.load("../train_equivariant_cloro/all_datasets/filters_names_bins/name_ligand.npy")

    Bins_list_old=np.load("../train_equivariant_cloro/all_datasets/filters_names_bins/bins_ligand.npy")
    #Bins_list_ligand_old=np.load("../dataset_latent_space_MUV/bins_ligand.npy")

    Bins_list=np.zeros((9,num_bins+1))
    for i in range(Bins_list.shape[0]):
        for j in range(Bins_list.shape[1]):
            Bins_list[i,j]=Bins_list_old[i,j*int(40/num_bins)]

    Matrix_lig=np.zeros((len(Name_list_ligand),num_bins+2))
    for j,atom in enumerate(latent_vec_ligand):
        for w in range(len(Name_list_ligand)):
            if atoms_ligand[j]==Name_list_ligand[w]:
                index=int(np.digitize(latent_vec_ligand[j], Bins_list[w]))
                Matrix_lig[w,index]+=1
    return Matrix_lig, Name_list_ligand, Bins_list
    
    
def Get_encoded_matrix_pocket(latent_vec_pocket, atoms_pocket, num_bins=40):

    #Name_list=np.load("../dataset_latent_space_MUV/name_pocket.npy")
    Name_list_pocket=np.load("../train_equivariant_cloro/all_datasets/filters_names_bins/name_pocket.npy")

    Bins_list_old=np.load("../train_equivariant_cloro/all_datasets/filters_names_bins/bins_pocket.npy")
    #Bins_list_ligand_old=np.load("../dataset_latent_space_MUV/bins_ligand.npy")

    Bins_list=np.zeros((6,num_bins+1))
    for i in range(Bins_list.shape[0]):
        for j in range(Bins_list.shape[1]):
            Bins_list[i,j]=Bins_list_old[i,j*int(40/num_bins)]

    Matrix_pock=np.zeros((len(Name_list_pocket),num_bins+2))
    for j,atom in enumerate(latent_vec_pocket):
        for w in range(len(Name_list_pocket)):
            if atoms_pocket[j]==Name_list_pocket[w]:
                index=int(np.digitize(latent_vec_pocket[j], Bins_list[w]))
                Matrix_pock[w,index]+=1
    return Matrix_pock, Name_list_pocket, Bins_list
                
                
