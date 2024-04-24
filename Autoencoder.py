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


class Autoencoder_cmap_INV(torch.nn.Module):
    def __init__(self, SN_cmap, max_n_atoms, num_atoms_type, batch_size, activation_func_mlp_cmap, n_nodes_mlp_cmap, n_layers_mlp_cmap, dim_latent, device):
        super(Autoencoder_cmap_INV, self).__init__()
        self.SN_cmap = SN_cmap

        modules = []
        if activation_func_mlp_cmap=="ELU":
            modules.append(torch.nn.Linear(dim_latent, n_nodes_mlp_cmap))
            modules.append(torch.nn.ELU())
        if activation_func_mlp_cmap=="Tanh":
            modules.append(torch.nn.Linear(dim_latent, n_nodes_mlp_cmap))
            modules.append(torch.nn.Tanh())
        for i in range(n_layers_mlp_cmap-1):
            if activation_func_mlp_cmap=="ELU":
                modules.append(torch.nn.Linear(n_nodes_mlp_cmap, n_nodes_mlp_cmap))
                modules.append(torch.nn.ELU())
            if activation_func_mlp_cmap=="Tanh":
                modules.append(torch.nn.Linear(n_nodes_mlp_cmap, n_nodes_mlp_cmap))
                modules.append(torch.nn.Tanh())
        modules.append(torch.nn.Linear(n_nodes_mlp_cmap,int(max_n_atoms*(max_n_atoms+1)/2)))
        modules.append(torch.nn.Sigmoid())

        self.mlp_cmap = torch.nn.Sequential(*modules)
        
        self.num_atoms_type = num_atoms_type
        
        self.max_n_atoms = max_n_atoms
        self.device = device
        self.batch_size = batch_size
        self.dim_latent = dim_latent
        
    def forward(self, item,batch_size):
        x_cmap = self.SN_cmap(item)
        #my_pad = torch.zeros(batch_size, self.dim_latent).to(self.device)
        #print(x_cmap.size())
        #for i in range(batch_size):
        #    my_pad[i,0:item["ptr"][i+1]-item["ptr"][i]] = x_cmap[item["ptr"][i]: item["ptr"][i+1]].squeeze()
        #x1_diag = self.mlp_cmap(my_pad)
        x1_diag = 3*self.mlp_cmap(x_cmap)
        x1 = torch.zeros(batch_size, self.max_n_atoms, self.max_n_atoms).to(self.device)
        ii, jj = torch.triu_indices(self.max_n_atoms, self.max_n_atoms)
        for i in range(batch_size):
            x1[i,ii,jj]= x1_diag[i]
            x1[i].T[ii,jj]= x1_diag[i]
            
        
        return x1.reshape(batch_size,self.max_n_atoms*self.max_n_atoms)



class Autoencoder_cmap_Pool(torch.nn.Module):
    def __init__(self, SN_cmap, max_n_atoms, num_atoms_type, batch_size, activation_func_mlp_cmap, n_nodes_mlp_cmap, n_layers_mlp_cmap, features_out, device):
        super(Autoencoder_cmap, self).__init__()
        self.SN_cmap = SN_cmap

        modules = []
        if activation_func_mlp_cmap=="ELU":
            modules.append(torch.nn.Linear(features_out, n_nodes_mlp_cmap))
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

    def forward(self, item):
        x_cmap = self.SN_cmap(item)
        x1_diag = self.mlp_cmap(x_cmap)

        x1 = torch.zeros(self.batch_size, self.max_n_atoms, self.max_n_atoms).to(self.device)
        ii, jj = torch.triu_indices(self.max_n_atoms, self.max_n_atoms)
        for i in range(self.batch_size):
            x1[i,ii,jj]= x1_diag[i]
            x1[i].T[ii,jj]= x1_diag[i]


        return x1.reshape(self.batch_size,self.max_n_atoms*self.max_n_atoms)




class RSN_with_label(torch.nn.Module):
    def __init__(self, SN_cmap, SN_atoms, max_n_atoms, num_atoms_type, batch_size, activation_func_mlp_cmap, activation_func_mlp_atoms, n_nodes_mlp_cmap, n_nodes_mlp_atoms, n_layers_mlp_cmap, n_layers_mlp_atoms, device):
        super(RSN_with_label, self).__init__()
        self.SN_cmap = SN_cmap
        self.SN_atoms= SN_atoms

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
        
        
        modules = []
        if activation_func_mlp_atoms=="ELU":
            modules.append(torch.nn.Linear(max_n_atoms, n_nodes_mlp_atoms))
            modules.append(torch.nn.ELU())
        if activation_func_mlp_atoms=="Tanh":
            modules.append(torch.nn.Linear(max_n_atoms, n_nodes_mlp_atoms))
            modules.append(torch.nn.Tanh())
        for i in range(n_layers_mlp_atoms-1):
            if activation_func_mlp_atoms=="ELU":
                modules.append(torch.nn.Linear(n_nodes_mlp_atoms, n_nodes_mlp_atoms))
                modules.append(torch.nn.ELU())
            if activation_func_mlp_atoms=="Tanh":
                modules.append(torch.nn.Linear(n_nodes_mlp_atoms, n_nodes_mlp_atoms))
                modules.append(torch.nn.Tanh())
        modules.append(torch.nn.Linear(n_nodes_mlp_atoms,max_n_atoms*(num_atoms_type+1)))
        #modules.append(torch.nn.Softmax())

        self.mlp_atoms = torch.nn.Sequential(*modules)
        
        self.num_atoms_type = num_atoms_type
        
        self.max_n_atoms = max_n_atoms
        self.device = device
        self.batch_size = batch_size
        
    def forward(self, item):
        x_cmap = self.SN_cmap(item)
        my_pad = torch.zeros(self.batch_size, self.max_n_atoms).to(self.device)
        for i in range(self.batch_size):
            my_pad[i,0:item["ptr"][i+1]-item["ptr"][i]] = x_cmap[item["ptr"][i]: item["ptr"][i+1]].squeeze()
        x1_diag = self.mlp_cmap(my_pad)
        
        x1 = torch.zeros(self.batch_size, self.max_n_atoms, self.max_n_atoms).to(self.device)
        ii, jj = torch.triu_indices(self.max_n_atoms, self.max_n_atoms)
        for i in range(self.batch_size):
            x1[i,ii,jj]= x1_diag[i]
            x1[i].T[ii,jj]= x1_diag[i]
            
            
        x_atoms = self.SN_atoms(item)
        my_pad_atoms = torch.zeros(self.batch_size, self.max_n_atoms).to(self.device)
        for i in range(self.batch_size):
            my_pad_atoms[i,0:item["ptr"][i+1]-item["ptr"][i]] = x_atoms[item["ptr"][i]: item["ptr"][i+1]].squeeze()
            
        x2 = self.mlp_atoms(my_pad_atoms)
        
        return x1.reshape(self.batch_size,self.max_n_atoms*self.max_n_atoms), x2





class RSN_insto(torch.nn.Module):
    def __init__(self, SN_cmap, SN_atoms, max_n_atoms, num_atoms_type, batch_size, activation_func_mlp_cmap, activation_func_mlp_atoms, n_nodes_mlp_cmap, n_nodes_mlp_atoms, n_layers_mlp_cmap, n_layers_mlp_atoms, device):
        super(RSN_insto, self).__init__()
        self.SN_cmap = SN_cmap
        self.SN_atoms= SN_atoms
        self.bin = Calculate_bins(max_n_atoms, device)
        self.sigma = 1/(max_n_atoms)

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
        
        
        modules = []
        if activation_func_mlp_atoms=="ELU":
            modules.append(torch.nn.Linear(max_n_atoms, n_nodes_mlp_atoms))
            modules.append(torch.nn.ELU())
        if activation_func_mlp_atoms=="Tanh":
            modules.append(torch.nn.Linear(max_n_atoms, n_nodes_mlp_atoms))
            modules.append(torch.nn.Tanh())
        for i in range(n_layers_mlp_atoms-1):
            if activation_func_mlp_atoms=="ELU":
                modules.append(torch.nn.Linear(n_nodes_mlp_atoms, n_nodes_mlp_atoms))
                modules.append(torch.nn.ELU())
            if activation_func_mlp_atoms=="Tanh":
                modules.append(torch.nn.Linear(n_nodes_mlp_atoms, n_nodes_mlp_atoms))
                modules.append(torch.nn.Tanh())
        modules.append(torch.nn.Linear(n_nodes_mlp_atoms,max_n_atoms*(num_atoms_type+1)))
        #modules.append(torch.nn.Softmax())

        self.mlp_atoms = torch.nn.Sequential(*modules)
        
        self.num_atoms_type = num_atoms_type
        
        self.max_n_atoms = max_n_atoms
        self.device = device
        self.batch_size = batch_size
        
    def forward(self, item):
        x_cmap = self.SN_cmap(item)
        my_pad = torch.zeros(self.batch_size, self.max_n_atoms).to(self.device)
        for i in range(self.batch_size):
            my_vec=x_cmap[item["ptr"][i]: item["ptr"][i+1]]
            my_vec_norm=(my_vec-torch.min(my_vec))*2/(torch.max(my_vec)-torch.min(my_vec)) -1
            my_pad[i,0:item["ptr"][i+1]-item["ptr"][i]] = self.Histo_Gaus(my_vec_norm.squeeze(), self.device)
        
        x1_diag = self.mlp_cmap(my_pad)
        
        x1 = torch.zeros(self.batch_size, self.max_n_atoms, self.max_n_atoms).to(self.device)
        ii, jj = torch.triu_indices(self.max_n_atoms, self.max_n_atoms)
        for i in range(self.batch_size):
            x1[i,ii,jj]= x1_diag[i]
            x1[i].T[ii,jj]= x1_diag[i]
            
            
        x_atoms = self.SN_atoms(item)
        my_pad_atoms = torch.zeros(self.batch_size, self.max_n_atoms).to(self.device)
        for i in range(self.batch_size):
            my_vec=x_atoms[item["ptr"][i]: item["ptr"][i+1]]
            my_vec_norm=(my_vec-torch.min(my_vec))*2/(torch.max(my_vec)-torch.min(my_vec)) -1
            my_pad_atoms[i,0:item["ptr"][i+1]-item["ptr"][i]] = self.Histo_Gaus(my_vec_norm.squeeze(), self.device)
            
        x2 = self.mlp_atoms(my_pad_atoms)
        
        return x1.reshape(self.batch_size,self.max_n_atoms*self.max_n_atoms), x2
        
        
    def Histo_Gaus(self,my_vec_norm, device):
        y=torch.zeros(len(my_vec_norm)).to(device)
        y=torch.exp(-(my_vec_norm-self.bin.unsqueeze(1))**2/(2*self.sigma**2)).sum(dim=1)
        return y

def Calculate_bins(max_n_atoms, device):
    a=torch.rand((5))
    r=(a-torch.min(a))*2/(torch.max(a)-torch.min(a)) -1
    bins=  torch.histogram(r,bins=max_n_atoms)[1]+(torch.histogram(r,bins=20)[1][-1]-torch.histogram(r,bins=20)[1][-2])*0.5
    return bins[1:].to(device)
