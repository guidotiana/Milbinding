
import numpy as np
import os
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
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
from Autoencoder import RSN_with_label, RSN_insto, Autoencoder_cmap
import torch
import mdtraj as md

import warnings
warnings.filterwarnings('ignore')

from utils_reproduce import fit_function, fit_function1, fit_function2, FeedforwardNeuralNetModel, KL_symm, get_ligand_from_name


### Geometrical Isometry ###

print("------------------------------------------------------------")
print("Evaluating the correlation between the encoding matrix")
print("and the dRMSD for different conformations of the ")
print("pocket of 2FL2 protein obtained via MD simulations")

Name_list=np.load("./all_datasets/filters_names_bins/name_pocket.npy")
Bins_list_old=np.load("./all_datasets/filters_names_bins/bins_pocket.npy")
latent_vec_pocket=np.load("./all_datasets/2FL2_md/latent_vec_pocket.npy")
pocket_atoms=np.load("./all_datasets/2FL2_md/pocket_atoms.npy")
real_diff=np.load("./all_datasets/2FL2_md/real_diff.npy")

num_bins=2


Bins_list=np.zeros((6,3))
for i in range(Bins_list.shape[0]):
    for j in range(Bins_list.shape[1]):
        Bins_list[i,j]=Bins_list_old[i,j*20]



Bins_list_values=np.zeros((Bins_list.shape[0],Bins_list.shape[1]+1))
for i in range(Bins_list_values.shape[0]):
    for j in range(Bins_list_values.shape[1]):
        if j==0:
            Bins_list_values[i,j]=Bins_list[i,j]-0.5*abs(Bins_list[i,1]-Bins_list[i,1])
        else:
            Bins_list_values[i,j]=Bins_list[i,j-1]+0.5*abs(Bins_list[i,1]-Bins_list[i,0])

print("Creating Matrix for Pockets train")
train_data_Matrix_Histo_Pocket=np.zeros((len(latent_vec_pocket),len(Name_list),num_bins+2))

for i,atom_types in enumerate(latent_vec_pocket):
    for j,atom in enumerate(pocket_atoms):
        for w in range(len(Name_list)):
            if pocket_atoms[j]==Name_list[w]:
                index=int(np.digitize(latent_vec_pocket[i][j], Bins_list[w]))
                train_data_Matrix_Histo_Pocket[i,w,index]+=1


matrix_diff=[]
dMs=[]

for i in range(500):
    mat_val=KL_symm(train_data_Matrix_Histo_Pocket[i].reshape(-1), train_data_Matrix_Histo_Pocket[-1].reshape(-1))
    matrix_diff.append(mat_val)



plt.figure(figsize=(8, 6))
sns.reset_orig()

params, cov = optimize.curve_fit(fit_function, np.array(matrix_diff).flatten(), np.array(real_diff), maxfev=8000)
y_fit = fit_function(np.sort(np.array(matrix_diff).flatten()), params[0], params[1])
std_dev = np.sqrt(np.diag(cov))


#plt.title('2FL2 Trajectory', fontsize=18)

plt.plot(np.sort(np.array(matrix_diff).flatten()), y_fit, c="tab:orange", label='Fit', alpha=0.8)
plt.fill_between(np.sort(np.array(matrix_diff).flatten()), y_fit - std_dev[0], y_fit + std_dev[0], color='tab:orange', alpha=0.2, label='Standard deviation')
plt.scatter(np.array(matrix_diff).flatten(), np.array(real_diff), c='tab:blue', s=50, alpha=0.7, label='Data')
plt.text(0.8, 0.1, r'$\rho$ = {:.2f}'.format(sp.stats.pearsonr(np.array(matrix_diff).flatten(), np.array(real_diff))[0]),
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=22)

plt.ylim(0.25,0.55)
# Add gridlines
plt.grid(True, alpha=0.75)


# Set the xticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(r"$D_{JS}}$", fontsize=20)
plt.ylabel(r"$\Delta$ dRMSD  [Å]", fontsize=20)

plt.legend(fontsize=15)
plt.savefig("./images_results/Geometrical_iso")




print("------------------------------------------------------------")
print("Evaluating the correlation between the encoding matrix")
print("and the Molecular Weight, Volume and Complexity of ligands")

np.random.seed(seed=1)

Name_list=np.load("./all_datasets/filters_names_bins/name_ligand.npy")
Bins_list=np.load("./all_datasets/filters_names_bins/bins_ligand.npy")
latent_vectors_all=np.load("./all_datasets/MUV/Properties/latent_vectors_all.npy",allow_pickle=True)
names_atoms_all=np.load("./all_datasets/MUV/Properties/names_atoms_all.npy",allow_pickle=True)
dict_prop=np.load("./all_datasets/MUV/Properties/dict_prop.npy",allow_pickle=True).item()

num_bins=40



Bins_list_values=np.zeros((Bins_list.shape[0],Bins_list.shape[1]+1))
for i in range(Bins_list_values.shape[0]):
    for j in range(Bins_list_values.shape[1]):
        if j==0:
            Bins_list_values[i,j]=Bins_list[i,j]-0.5*abs(Bins_list[i,1]-Bins_list[i,1])
        else:
            Bins_list_values[i,j]=Bins_list[i,j-1]+0.5*abs(Bins_list[i,1]-Bins_list[i,0])

Matrix_Histo_Ligand=np.zeros((len(latent_vectors_all),len(Name_list),num_bins+2))

for i,atom_types in enumerate(latent_vectors_all):
    for j,atom in enumerate(names_atoms_all[i]):
        for w in range(len(Name_list)):
            if names_atoms_all[i][j]==Name_list[w]:
                index=int(np.digitize(latent_vectors_all[i][j], Bins_list[w]))
                Matrix_Histo_Ligand[i,w,index]+=1
                
norm_matrixes=[]
for i in range(len(Matrix_Histo_Ligand)):
    norm_matrixes.append((np.sum((Matrix_Histo_Ligand[i]*Bins_list_values)**2))**0.5)
    
    



for prop in dict_prop:
    if prop=="MolecularWeight" or prop=="Volume3D" or prop=="Complexity":
        sns.set_theme()
        sns.reset_orig()
        plt.figure()
        sns.scatterplot(dict_prop[prop], norm_matrixes,edgecolors="face", alpha=0.7)

        plt.ylabel(r"$\lambda_M$", fontsize=20)
        if prop=="MolecularWeight":
            plt.xlabel(r"MW [$g/mol$]", fontsize=15)
            params=optimize.curve_fit(fit_function1, dict_prop[prop], norm_matrixes, maxfev=8000)
            y_fit=fit_function1(dict_prop[prop], params[0][0], params[0][1])
        if prop=="Volume3D":
            plt.xlabel(r"V [$\AA ^3$]", fontsize=15)
            params=optimize.curve_fit(fit_function1, dict_prop[prop], norm_matrixes, maxfev=8000)
            y_fit=fit_function1(dict_prop[prop], params[0][0], params[0][1])
        if prop=="Complexity":
            plt.xlabel("Complexity")
            params=optimize.curve_fit(fit_function2, dict_prop[prop], norm_matrixes, p0=(180,0,1), maxfev=8000)
            y_fit=fit_function2(dict_prop[prop], params[0][0], params[0][1], params[0][2])
        print(f"Reference {prop} = {params[0][0]}")

        
        ss_res = np.sum((norm_matrixes - y_fit) ** 2)
        ss_tot = np.sum((norm_matrixes - np.mean(norm_matrixes)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        x=np.arange(1000)
        plt.yscale('log')
        plt.grid()
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        #plt.savefig(f"./images/AAAA_MAT_NORM_{prop}", dpi=900)
        print("r^2 = "+str(round(r2, 5)))
        plt.savefig(f"./images_results/Prop_{prop}")


# In[33]:


### NN for predicting properties XLogP ###


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"


    
print("----------------------------------------------------------------------------------------")
print("Evaluating trained (shallow) Neural Network for predicting XlogP")
    
matrixes=np.load("./all_datasets/MUV/Properties/matrixes_for_regression.npy")
XlogPs = np.load("./all_datasets/MUV/Properties/xlogp_for_regression.npy")
polarareas = np.load("./all_datasets/MUV/Properties/polararea_for_regression.npy")

indixes=np.arange(len(matrixes))
np.random.shuffle(indixes)

matrixes_train=[]
matrixes_val=[]
XlogPs_train=[]
XlogPs_val=[]
XlogPs_all=[]

test_size=0.85
epochs=250

tot_run=0

for i in range(int(len(indixes)*test_size)):
    if math.isnan(XlogPs[indixes[i]]) != True :
        matrixes_train.append(matrixes[indixes[i]])
        XlogPs_train.append(XlogPs[indixes[i]])
        XlogPs_all.append(XlogPs[indixes[i]])
    
for i in range(int(len(indixes)*test_size)+1, len(indixes)):
    if math.isnan(XlogPs[indixes[i]]) != True :
        matrixes_val.append(matrixes[indixes[i]])
        XlogPs_val.append(XlogPs[indixes[i]])
        XlogPs_all.append(XlogPs[indixes[i]])
        
count=len(matrixes_train)+len(matrixes_val)


train_load = TensorDataset(torch.tensor(np.array(matrixes_train).astype(np.float32)).to(device),torch.tensor(np.array(XlogPs_train).astype(np.float32)).to(device) )
train_set = DataLoader(train_load, batch_size=1, shuffle=True)

val_load = TensorDataset(torch.tensor(np.array(matrixes_val).astype(np.float32)),torch.tensor(np.array(XlogPs_val).astype(np.float32)) )
val_set = DataLoader(val_load, batch_size=1, shuffle=True)


net = FeedforwardNeuralNetModel(9*42,10,2,1, "Linear", 0.0)
net.load_state_dict(torch.load("./all_datasets/MUV/Properties/regressor_models/xlogp/nn_val_xlogp.pt"))
net.to(device)
net.eval()
loss = nn.MSELoss(reduction='mean')


loss_train=0
train_loss_epochs=[]
loss_val=0
val_loss_epochs=[]

val_pre=0
train_pre=0

tempo_train_loss=[]
tempo_val_loss=[]
tot_i_t=0


for batch_idx, batch in enumerate(train_set):
    output = net(batch[0].to(device)).reshape(-1)
    loss_train = float((loss(output.reshape(-1),batch[1].to(device).reshape(-1))**0.5)/(batch[1].to(device).reshape(-1)))
    #tempo_train_loss.append(loss_train.cpu().detach().numpy())
    if loss_train!= float("inf"):
        tempo_train_loss.append(loss_train)
    
for batch_idx, batch in enumerate(val_set):
    output = net(batch[0].to(device))
    loss_val = float((loss(output.reshape(-1),batch[1].to(device).reshape(-1))**0.5)/(batch[1].to(device).reshape(-1)))
    #tempo_val_loss.append(loss_val.cpu().detach().numpy())
    if loss_val!= float("inf"):
        tempo_val_loss.append(loss_val)                  

                            
fig, axs = plt.subplots(2, 1, layout='constrained')
plt.suptitle("XLogP")
axs[0].hist(tempo_train_loss,bins=100, density=True, range=(0,10), label="Train Set")
axs[0].set_xlabel('|target - predicted| / target')
#axs[0].set_ylabel('frequency')
axs[0].set_xlim(-1, 10)
axs[0].axvline(np.mean(tempo_train_loss), color='k', linestyle='dashed', linewidth=1.5)
axs[0].grid(True)
value=round(float(np.mean(tempo_train_loss)),3)
axs[0].text(0.8,1.35,fr"Mean = {value}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,fontsize=10)
axs[0].legend()
print(f"Training set : Mean[|target - predicted| / target] = {value}")

axs[1].hist(tempo_val_loss,bins=100, density=True, range=(0,10), color="tab:orange", label="Val Set")
axs[1].set_xlabel('|target - predicted| / target')
#axs[1].set_ylabel('frequency')
axs[1].set_xlim(-1, 10)
axs[1].axvline(np.mean(tempo_val_loss), color='k', linestyle='dashed', linewidth=1.5)
axs[1].grid(True)
axs[1].legend()
value=round(float(np.mean(tempo_val_loss)),3)
axs[1].text(0.8,0.1,fr"Mean = {value}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,fontsize=10)
        
plt.savefig("./images_results/Prop_xlogp")
print(f"Validation set : Mean[|target - predicted| / target] = {value}")


# In[32]:


print("----------------------------------------------------------------------------------------")
print("Evaluating trained (shallow) Neural Network for predicting Polar Surface Area")

matrixes=np.load("./all_datasets/MUV/Properties/matrixes_for_regression.npy")
polarareas = np.load("./all_datasets/MUV/Properties/polararea_for_regression.npy")

indixes=np.arange(len(matrixes))
np.random.shuffle(indixes)

matrixes_train=[]
matrixes_val=[]
polarareas_train=[]
polarareas_val=[]
polarareas_all=[]

test_size=0.85
epochs=250

tot_run=0

for i in range(int(len(indixes)*test_size)):
    if math.isnan(polarareas[indixes[i]]) != True and float(polarareas[indixes[i]])>=50.0 :
        matrixes_train.append(matrixes[indixes[i]])
        polarareas_train.append(polarareas[indixes[i]])
        polarareas_all.append(polarareas[indixes[i]])
    
for i in range(int(len(indixes)*test_size)+1, len(indixes)):
    if math.isnan(polarareas[indixes[i]]) != True and float(polarareas[indixes[i]])>=50.0 :
        matrixes_val.append(matrixes[indixes[i]])
        polarareas_val.append(polarareas[indixes[i]])
        polarareas_all.append(polarareas[indixes[i]])
        
count=len(matrixes_train)+len(matrixes_val)
#print("Dimension of the Dataset")
#print(f"Number of matrixes = {count}")

train_load = TensorDataset(torch.tensor(np.array(matrixes_train).astype(np.float32)).to(device),torch.tensor(np.array(polarareas_train).astype(np.float32)).to(device) )
train_set = DataLoader(train_load, batch_size=1, shuffle=True)

val_load = TensorDataset(torch.tensor(np.array(matrixes_val).astype(np.float32)),torch.tensor(np.array(polarareas_val).astype(np.float32)) )
val_set = DataLoader(val_load, batch_size=1, shuffle=True)


net = FeedforwardNeuralNetModel(9*42,10,1,1, "ELU", 0.0)
net.load_state_dict(torch.load("./all_datasets/MUV/Properties/regressor_models/polararea/nn_val_polararea.pt"))
net.to(device)
net.eval()
loss = nn.MSELoss(reduction='mean')


loss_train=0
train_loss_epochs=[]
loss_val=0
val_loss_epochs=[]

val_pre=0
train_pre=0

tempo_train_loss=[]
tempo_val_loss=[]
tot_i_t=0


for batch_idx, batch in enumerate(train_set):
    output = net(batch[0].to(device))
    loss_train = abs(float(output[0])-float(batch[1][0]))/float(batch[1][0])
    #loss_train = float((loss(output.reshape(-1),batch[1].to(device).reshape(-1))**0.5)/(batch[1].to(device).reshape(-1)))
    if loss_train!=float("inf"):
        tempo_train_loss.append(loss_train)
                            
for batch_idx, batch in enumerate(val_set):
    output = net(batch[0].to(device))
    loss_val = abs(float(output[0])-float(batch[1][0]))/float(batch[1][0])
    #loss_val = float((loss(output.reshape(-1),batch[1].to(device).reshape(-1))**0.5)/(batch[1].to(device).reshape(-1)))
    if loss_val!=float("inf"):
        tempo_val_loss.append(loss_val)
        
fig, axs = plt.subplots(2, 1, layout='constrained')
plt.suptitle("Polar Surface Area")
axs[0].hist(tempo_train_loss,bins=100, density=True, range=(0,5), label="Train Set")
axs[0].set_xlabel('|target - predicted| / target')
#axs[0].set_ylabel('frequency')
axs[0].set_xlim(-1, 5)
axs[0].axvline(np.mean(tempo_train_loss), color='k', linestyle='dashed', linewidth=1.5)
axs[0].grid(True)
value=round(float(np.mean(tempo_train_loss)),3)
axs[0].text(0.8,1.35,fr"Mean = {value}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,fontsize=10)
axs[0].legend()
print(f"Training set : Mean[|target - predicted| / target] = {value}")

axs[1].hist(tempo_val_loss,bins=100, density=True, range=(0,5), color="tab:orange", label="Val Set")
axs[1].set_xlabel('|target - predicted| / target')
#axs[1].set_ylabel('frequency')
axs[1].set_xlim(-1, 5)
axs[1].axvline(np.mean(tempo_val_loss), color='k', linestyle='dashed', linewidth=1.5)
axs[1].grid(True)
axs[1].legend()
value=round(float(np.mean(tempo_val_loss)),3)
axs[1].text(0.8,0.1,fr"Mean = {value}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,fontsize=10)
plt.savefig("./images_results/Prop_polar_surf_area")
print(f"Validation set : Mean[|target - predicted| / target] = {value}")



# In[3]:


##### Train DUDE Test on MUV ####

print("------------------------------------------------------------")
print("Evaluating the AUC score value for RF trained on DUDE")
print("and tested on MUV")


train_dataset_MUV=np.load("./all_datasets/MUV/training_matrix_MUV_10_bins.npz")
train_pock_MUV=train_dataset_MUV["pocket"]
train_dataset_MUV=np.load("./all_datasets/MUV/training_matrix.npz")
train_lig_MUV=train_dataset_MUV["ligand"]

target_train_MUV=np.load("./all_datasets/MUV/target_train.npy")

X_MUV=[]
y_MUV=[]

for i in range(len(train_pock_MUV)):
    X_MUV.append(np.concatenate((train_pock_MUV[i].reshape(-1), train_lig_MUV[i].reshape(-1))).reshape(-1))
    y_MUV.append(target_train_MUV[i])


val_dataset_MUV=np.load("./all_datasets/MUV/validation_matrix_MUV_10_bins.npz")
val_pock_MUV=val_dataset_MUV["pocket"]
val_dataset_MUV=np.load("./all_datasets/MUV/validation_matrix.npz")
val_lig_MUV=val_dataset_MUV["ligand"]

target_val_MUV=np.load("./all_datasets/MUV/target_val.npy")

for i in range(len(val_pock_MUV)):
    X_MUV.append(np.concatenate((val_pock_MUV[i].reshape(-1), val_lig_MUV[i].reshape(-1))).reshape(-1))
    y_MUV.append(target_val_MUV[i])
    
RF = RandomForestClassifier()

with open("./final_random_forests/Best_DUDE_on_MUV_10bins/rf_test_MUV", "rb") as f:
    RF = cPickle.load(f)
    
y_pred_MUV=RF.predict(X_MUV)
auc_MUV=roc_auc_score(y_MUV, y_pred_MUV)
print(f"AUC on MUV with RF trained on DUDE = {auc_MUV}")


# In[4]:


#### DUDE on DUDE

print("------------------------------------------------------------")
print("Evaluating the AUC score value for RF trained on DUDE")
print("and tested on DUDE (validation set)")


seed = 123
np.random.seed(seed)
random.seed(seed)


indexes=np.load("./all_datasets/DUDE/indexes_val_repetitions_DUDE.npy")

train_dataset=np.load("./all_datasets/DUDE/training_matrix.npz")
train_pock=train_dataset["pocket"]
train_lig=train_dataset["ligand"]

target_train=np.load("./all_datasets/DUDE/target_train.npy")

X_train=[]
y_train=[]

for i in range(len(train_pock)):
    X_train.append(np.concatenate((train_pock[i], train_lig[i])).reshape(-1))
    y_train.append(target_train[i])
    
    
val_dataset=np.load("./all_datasets/DUDE/validation_matrix.npz")
val_pock=val_dataset["pocket"]
val_lig=val_dataset["ligand"]

target_val=np.load("./all_datasets/DUDE/target_val.npy")

X_val=[]
y_val=[]
    
no_loaded=[]

stat=0

for i in range(len(val_pock)):
    if stat<len(indexes):
        if i==indexes[stat]:
            stat+=1
            no_loaded.append(i)
        else:
            X_val.append(np.concatenate((val_pock[i], val_lig[i])).reshape(-1))
            y_val.append(target_val[i])
            

with open("./final_random_forests/Best_DUDE_on_DUDE_40bins/rf_train_NoRep", "rb") as f:
    RF = cPickle.load(f)         
        
y_pred_train=RF.predict(X_train)
auc_train=roc_auc_score(y_train, y_pred_train)
            
with open("./final_random_forests/Best_DUDE_on_DUDE_40bins/rf_val_NoRep", "rb") as f:
    RF = cPickle.load(f)
    
y_pred_val=RF.predict(X_val)
auc_val=roc_auc_score(y_val, y_pred_val)

print(f"AUC on DUDE dataste train = {auc_train}; val = {auc_val}")

positions = (0.5, 1.5)
labels_x = ('0 real','1 real')
labels_y = ('0 pred','1 pred')

plt.figure()
mat = metrics.confusion_matrix(y_train, y_pred_train, normalize='true')
sns.heatmap(mat.T*100, annot=True,  cmap="viridis", fmt='.2f')
plt.title("DUDE training - RF", fontsize=15)
plt.xticks(positions, labels_x, rotation=45)
plt.yticks(positions, labels_y, rotation=45)
plt.savefig("./images_results/Auc_score_dude_train")

plt.figure()
mat = metrics.confusion_matrix(y_val, y_pred_val, normalize='true')
sns.heatmap(mat.T*100, annot=True,  cmap="viridis", fmt='.2f')
plt.title("DUDE validation - RF", fontsize=15)
plt.xticks(positions, labels_x, rotation=45)
plt.yticks(positions, labels_y, rotation=45)
plt.savefig("./images_results/Auc_score_dude_validation")



# In[6]:


#### MUV on MUV ####

print("------------------------------------------------------------")
print("Evaluating the AUC score value for RF trained on MUV")
print("and tested on MUV (validation set)")

indexes_MUV=np.load("./all_datasets/MUV/indexes_val_repetitions_MUV.npy")


train_dataset=np.load("./all_datasets/MUV/training_matrix.npz")
train_pock=train_dataset["pocket"]
train_lig=train_dataset["ligand"]

target_train=np.load("./all_datasets/MUV/target_train.npy")

X_train=[]
y_train=[]

for i in range(len(train_pock)):
    X_train.append(np.concatenate((train_pock[i], train_lig[i])).reshape(-1))
    y_train.append(target_train[i])
    
    
val_dataset=np.load("./all_datasets/MUV/validation_matrix.npz")
val_pock=val_dataset["pocket"]
val_lig=val_dataset["ligand"]

target_val=np.load("./all_datasets/MUV/target_val.npy")

X_val=[]
y_val=[]

no_loaded=[]

stat=0

for i in range(len(val_pock)):
    if stat < len(indexes_MUV):
        if i==indexes_MUV[stat]:
            stat+=1
            no_loaded.append(i)
        else:
            X_val.append(np.concatenate((val_pock[i], val_lig[i])).reshape(-1))
            y_val.append(target_val[i])
            

            
with open("./final_random_forests/Best_MUV_on_MUV_40bins/rf_train_NoRep", "rb") as f:
    RF = cPickle.load(f)
        
y_pred_train=RF.predict(X_train)
auc_train=roc_auc_score(y_train, y_pred_train)

with open("./final_random_forests/Best_MUV_on_MUV_40bins/rf_val_NoRep", "rb") as f:
    RF = cPickle.load(f)

y_pred_val=RF.predict(X_val)
auc_val=roc_auc_score(y_val, y_pred_val)

print(f"AUC on MUV dataste train = {auc_train}; val = {auc_val}")

positions = (0.5, 1.5)
labels_x = ('0 real','1 real')
labels_y = ('0 pred','1 pred')

plt.figure()
mat = metrics.confusion_matrix(y_train, y_pred_train, normalize='true')
sns.heatmap(mat.T*100, annot=True,  cmap="viridis", fmt='.2f')
plt.title("MUV training - RF", fontsize=15)
plt.xticks(positions, labels_x, rotation=45)
plt.yticks(positions, labels_y, rotation=45)
plt.savefig("./images_results/Auc_score_muv_validation")

plt.figure()
mat = metrics.confusion_matrix(y_val, y_pred_val, normalize='true')
sns.heatmap(mat.T*100, annot=True,  cmap="viridis", fmt='.2f')
plt.title("MUV validation - RF", fontsize=15)
plt.xticks(positions, labels_x, rotation=45)
plt.yticks(positions, labels_y, rotation=45)
plt.savefig("./images_results/Auc_score_muv_validation")


# In[23]:


#### Screen ####

print("------------------------------------------------------------")
print("Screening pocket 5exm from MUV with pockets from MUV")








rf = RandomForestClassifier()

with open('./final_random_forests/Best_MUV_on_MUV_40bins/rf_val_NoRep', 'rb') as f:
    rf = cPickle.load(f)
    
device='cpu'
matrixes_pock_val=np.load("./all_datasets/MUV/validation_matrix.npz")["pocket"]
matrixes_pock_train=np.load("./all_datasets/MUV/training_matrix.npz")["pocket"]
matrixes_lig_val=np.load("./all_datasets/MUV/validation_matrix.npz")["ligand"]
matrixes_lig_train=np.load("./all_datasets/MUV/training_matrix.npz")["ligand"]
target_val=np.load("./all_datasets/MUV/target_val.npy")

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
net_r_ligand.load_state_dict(torch.load(os.path.join("./equivariant_autoencoder",min_val_filename_ligand,"model_val.pt"), map_location=torch.device(device)))
my_set_for_filter_ligand=np.load("./all_datasets/filters_names_bins/my_set_for_filter_lig.npy")
my_filter_ligand=np.load("./all_datasets/filters_names_bins/my_filter_lig.npy", allow_pickle=True)
Name_list=np.load("./all_datasets/filters_names_bins/name_ligand.npy")
Bins_list=np.load("./all_datasets/filters_names_bins/bins_ligand.npy")
num_bins=40


Muv_pocket=np.load("./all_datasets/MUV/Muv_pockets.npy",allow_pickle=True).item()

pock_init=0
pock_finish=3000


min_percents=[]
names_per=[]

for name_pock in Muv_pocket:
    if name_pock=="5exm":
        matrixes_pock_val=Muv_pocket[name_pock]
        target_lig=get_ligand_from_name(name_pock)
        lig_active=[]
        lig_decoys=[]
        print("Start Screening")
        for j in range(len(matrixes_lig_val)):
            point=np.concatenate((matrixes_pock_val, matrixes_lig_val[j])).reshape(-1)
            if int(rf.predict(point.reshape(1,-1))) ==1:
                lig_active.append(matrixes_lig_val[j])
            else:
                lig_decoys.append(matrixes_lig_val[j])
        for j in range(len(matrixes_lig_train)):
            point=np.concatenate((matrixes_pock_val, matrixes_lig_train[j])).reshape(-1)
            if int(rf.predict(point.reshape(1,-1))) ==1:
                lig_active.append(matrixes_lig_train[j])
            else:
                lig_decoys.append(matrixes_lig_train[j])




        KL_pos=[]
        KL_neg=[]

        for j in range(len(lig_decoys)):
            KL_neg.append(KL_symm(target_lig.reshape(-1), lig_decoys[j].reshape(-1)))
        for j in range(len(lig_active)):
            KL_pos.append(KL_symm(target_lig.reshape(-1), lig_active[j].reshape(-1)))

        plt.figure()
        plt.hist(KL_pos,alpha=0.5, density=True, label="Positive",bins=30)
        plt.hist(KL_neg,alpha=0.5, density=True, label="Negative",bins=30)
        #plt.plot(xx, kde(xx), alpha=0.5, label="Benchmark")
        plt.title(r"$D_{JS}$",fontsize=18)
        plt.xlabel(r"$D_{JS}$",fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)


        plt.legend(fontsize=15)
        plt.grid(alpha=0.5)
        plt.savefig("./images_results/Screening_5exm")

        min_percents.append(len(lig_active)/(len(lig_active)+len(lig_decoys)))
        names_per.append(name_pock)


# In[36]:


##### Train DUDE Test on BindingDB ####
print("------------------------------------------------------------")
print("Evaluating the True positive rate for RF trained on DUDE")
print("and tested on PDBbind")


num_bins=4

train_dataset_BindingDB=np.load("./all_datasets/Binding_DB/training_matrix_BindingDB_4_bins.npz")
train_pock_BindingDB=train_dataset_BindingDB["pocket"]
train_dataset_BindingDB=np.load("./all_datasets/Binding_DB/training_matrix.npz")

train_lig_BindingDB=train_dataset_BindingDB["ligand"]

#train_lig_name_pdb=np.load("../dataset_latent_space_BindingDB/ligand_name_train.npy")

#target_train_DUDE=np.load("../dataset_latent_space_BindingDB/target_train.npy")

X_train_BindingDB=[]
#y_train_DUDE=[]
names_pdb_all=[]

for i in range(len(train_pock_BindingDB)):
    X_train_BindingDB.append(np.concatenate((train_pock_BindingDB[i].reshape(-1), train_lig_BindingDB[i].reshape(-1))).reshape(-1))
    #y_train_DUDE.append(target_train_DUDE[i])
    #names_pdb_all.append(train_lig_name_pdb[i])
    
    
val_dataset_BindingDB=np.load("./all_datasets/Binding_DB/validation_matrix_BindingDB_4_bins.npz")
val_pock_BindingDB=val_dataset_BindingDB["pocket"]
val_dataset_BindingDB=np.load("./all_datasets/Binding_DB/validation_matrix.npz")
val_lig_BindingDB=val_dataset_BindingDB["ligand"]
#val_lig_name_pdb=np.load("../dataset_latent_space_BindingDB/ligand_name_val.npy")

#target_val_DUDE=np.load("../dataset_latent_space_BindingDB/target_val.npy")

# I do like here because I do not need validation here

for i in range(len(val_pock_BindingDB)):
    X_train_BindingDB.append(np.concatenate((val_pock_BindingDB[i].reshape(-1), val_lig_BindingDB[i].reshape(-1))).reshape(-1))
    #y_train_DUDE.append(target_val_DUDE[i])
    #names_pdb_all.append(val_lig_name_pdb[i])
    
from sklearn import svm

RF = RandomForestClassifier()
#with open('./RF_NoRep_funziona/rf_val_NoRep', 'rb') as f:
with open("./final_random_forests/Best_DUDE_on_BindingDB_4bins/rf_val_NoRep", "rb") as f:
#with open('regressor_models/RF_NoRep/rf_val_NoRep', 'rb') as f:
    RF = cPickle.load(f)
    
y_pred_BindingDB=RF.predict(X_train_BindingDB)

Correct_values=0
False_values=0

names_incorrect=[]

for i,y in enumerate(y_pred_BindingDB):
    if y == 1:
        Correct_values+=1
    else:
        False_values+=1
        #names_incorrect.append(names_pdb_all[i])
        


#auc=roc_auc_score(y_train_DUDE, y_pred_DUDE)

print(f"Correct_values = {Correct_values}")
print(f"False_values = {False_values}")
print(f"True positive rate total = {Correct_values/(Correct_values+False_values)}")

from sklearn.metrics import confusion_matrix
confusion_matrix(np.ones_like(y_pred_BindingDB), y_pred_BindingDB)

my_filter_lig=np.load("./all_datasets/Binding_DB/my_filter_lig.npy",allow_pickle=True)
my_filter_lig.item()

atoms_mass={}
atoms_mass["C"]=12.011
atoms_mass["N"]=14.0067
atoms_mass["O"]=15.999
atoms_mass["S"]=32.065
atoms_mass["F"]=18.998403
atoms_mass["CL"]=35.453
atoms_mass["B"]=79.904
atoms_mass["I"]=126.90447
atoms_mass["P"]=30.973762




train_dataset_BindingDB=np.load("./all_datasets/Binding_DB/training_matrix_BindingDB_4_bins.npz")
train_pock_BindingDB=train_dataset_BindingDB["pocket"]
train_dataset_BindingDB=np.load("./all_datasets/Binding_DB/training_matrix.npz")
train_lig_BindingDB=train_dataset_BindingDB["ligand"]

train_lig_name_pdb=np.load("./all_datasets/Binding_DB/ligand_name_train.npy")

train_dataset_latent=np.load("./all_datasets/Binding_DB/train_dataset.npz")
name_train=train_dataset_latent["nam_ligand"]

#target_train_DUDE=np.load("../dataset_latent_space_BindingDB/target_train.npy")

X_train_BindingDB=[]
#y_train_DUDE=[]
names_pdb_all=[]

mass_all_ligand=[]

for i in range(len(train_pock_BindingDB)):
    mass_lig=0
    for atom in name_train[i]:
        if atom!="NULL":
            mass_lig+=atoms_mass[atom]
    mass_all_ligand.append(mass_lig)
    X_train_BindingDB.append(np.concatenate((train_pock_BindingDB[i].reshape(-1), train_lig_BindingDB[i].reshape(-1))).reshape(-1))

    #names_pdb_all.append(train_lig_name_pdb[i])
    
    
val_dataset_BindingDB=np.load("./all_datasets/Binding_DB/validation_matrix_BindingDB_4_bins.npz")
val_pock_BindingDB=val_dataset_BindingDB["pocket"]
val_dataset_BindingDB=np.load("./all_datasets/Binding_DB/validation_matrix.npz")
val_lig_BindingDB=val_dataset_BindingDB["ligand"]
val_lig_name_pdb=np.load("./all_datasets/Binding_DB/ligand_name_val.npy")


val_dataset_latent=np.load("./all_datasets/Binding_DB/val_dataset.npz")
name_val=val_dataset_latent["nam_ligand"]



for i in range(len(val_pock_BindingDB)):
    mass_lig=0
    for atom in name_val[i]:
        if atom!="NULL":
            mass_lig+=atoms_mass[atom]
    mass_all_ligand.append(mass_lig)

    X_train_BindingDB.append(np.concatenate((val_pock_BindingDB[i].reshape(-1), val_lig_BindingDB[i].reshape(-1))).reshape(-1))
    #names_pdb_all.append(val_lig_name_pdb[i])
    
from sklearn import svm

RF = RandomForestClassifier()
#with open('./RF_NoRep_funziona/rf_val_NoRep', 'rb') as f:
with open("./final_random_forests/Best_DUDE_on_BindingDB_4bins/rf_val_NoRep", "rb") as f:
#with open('regressor_models/RF_NoRep/rf_val_NoRep', 'rb') as f:
    RF = cPickle.load(f)
    
y_pred_BindingDB=RF.predict(X_train_BindingDB)

Correct_values=0
False_values=0

names_incorrect=[]

for i,y in enumerate(y_pred_BindingDB):
    if y == 1:
        Correct_values+=1
    else:
        False_values+=1
        #names_incorrect.append(names_pdb_all[i])

Freq_per_mass={}
Bins_for_mass=[70,140,210,280,350,420,491]
Freq_per_mass_neg={}

for m in range(len(Bins_for_mass)-1):
    Correct_values=0
    False_values=0
    for i,y in enumerate(y_pred_BindingDB):
        
        mass=mass_all_ligand[i]
        if m==len(Bins_for_mass)-1:
            if Bins_for_mass[m] < mass:
                print("HEY")
                if y == 1:
                    Correct_values+=1
                else:
                    False_values+=1
        else:
            if Bins_for_mass[m] < mass and mass < Bins_for_mass[m+1]:
                if y == 1:
                    Correct_values+=1
                else:
                    False_values+=1
    Freq_per_mass[str(m)]=Correct_values/(len(mass_all_ligand))
    Freq_per_mass_neg[str(m)]=False_values/len(mass_all_ligand)
    
import matplotlib.pyplot as plt


species=(
    "70~140",
    "140~210",
    "210~280",
    "280~350",
    "350~430",
    "430~490",
)

for i,key in enumerate(Freq_per_mass):
    print("---------------------------------------")
    print("True positive rates " +species[i]+" MW[u]")
    print(round(Freq_per_mass[key]/(Freq_per_mass_neg[key]+Freq_per_mass[key]),3))

weight_counts= {
    "Bind" : np.array([Freq_per_mass["0"],Freq_per_mass["1"],Freq_per_mass["2"],Freq_per_mass["3"],Freq_per_mass["4"],Freq_per_mass["5"]]),
    "Not Bind" : np.array([Freq_per_mass_neg["0"],Freq_per_mass_neg["1"],Freq_per_mass_neg["2"],Freq_per_mass_neg["3"],Freq_per_mass_neg["4"],Freq_per_mass_neg["5"]])

}

width = 0.25
#fig, ax=plt.subplots()
fig, ax = plt.subplots(figsize=(10, 6)) 
ax.grid(alpha=0.5)
bottom=np.zeros(6)
colurs=["forestgreen", "firebrick"]

stat=0

for boolean, weight_count in weight_counts.items():
    p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom, color=colurs[stat])
    bottom += weight_count
    stat+=1

#ax.set_title("Correct Bind prediction in PDBBind per weight", fontsize=15)

ax.set_xlabel("MW [u]", fontsize=18)
ax.set_ylabel("Positive rate", fontsize=18)

ax.legend(loc="upper left", fontsize=13)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#ax.set_ylim(0.0,1.5)
plt.savefig("./images_results/PDBbind_true_positiv_rate")


# In[ ]:




