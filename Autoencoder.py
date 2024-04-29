import torch
from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
import warnings
warnings.filterwarnings('ignore')

class Autoencoder_cmap(torch.nn.Module):
    def __init__(self, SN_cmap, max_n_atoms, num_atoms_type, batch_size, activation_func_mlp_cmap, n_nodes_mlp_cmap, n_layers_mlp_cmap, device):
        super(Autoencoder_cmap, self).__init__()
        self.SN_cmap = SN_cmap

        modules = []
        if activation_func_mlp_cmap=="ELU":
            modules.append(torch.nn.Linear(max_n_atoms, n_nodes_mlp_cmap))
            modules.append(torch.nn.ELU())
        if activation_func_mlp_cmap=="Tanh":
            modules.append(torch.nn.Linear(max_n_atoms, n_nodes_mlp_cmap))
            modules.append(torch.nn.Tanh())
        for i in range(n_layers_mlp_cmap-1):
            if activation_func_mlp_cmap=="ELU":
                modules.append(torch.nn.Linear(n_nodes_mlp_cmap, n_nodes_mlp_cmap))
                modules.append(torch.nn.ELU())
            if activation_func_mlp_cmap=="Tanh":
                modules.append(torch.nn.Linear(n_nodes_mlp_cmap, n_nodes_mlp_cmap))
                modules.append(torch.nn.Tanh())
        modules.append(torch.nn.Linear(n_nodes_mlp_cmap,int(max_n_atoms*(max_n_atoms+1)/2)))
        

        self.mlp_cmap = torch.nn.Sequential(*modules)
        
        self.num_atoms_type = num_atoms_type
        
        self.max_n_atoms = max_n_atoms
        self.device = device
        self.batch_size = batch_size
        
    def forward(self, item,batch_size):
        x_cmap = self.SN_cmap(item)
        my_pad = torch.zeros(batch_size, self.max_n_atoms).to(self.device)
        for i in range(batch_size):
            my_pad[i,0:item["ptr"][i+1]-item["ptr"][i]] = x_cmap[item["ptr"][i]: item["ptr"][i+1]].squeeze()
        x1_diag = self.mlp_cmap(my_pad)
        
        x1 = torch.zeros(batch_size, self.max_n_atoms, self.max_n_atoms).to(self.device)
        ii, jj = torch.triu_indices(self.max_n_atoms, self.max_n_atoms)
        for i in range(batch_size):
            x1[i,ii,jj]= x1_diag[i]
            x1[i].T[ii,jj]= x1_diag[i]
            
        
        return x1.reshape(batch_size,self.max_n_atoms*self.max_n_atoms)


