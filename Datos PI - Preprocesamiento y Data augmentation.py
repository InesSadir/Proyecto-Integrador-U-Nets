#!/usr/bin/env python
# coding: utf-8

# # Preprocesamiento: NIfTI a NumPy
# Antes de usar las imagenes hay que transformarlas a tensores numpy, ademas se las redimanciona para que tengan dimensiones 64x64x64.

# In[1]:


import os 
import glob
import torch 
import random
import numpy as np 
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate


# ## Crear carpetas
# 

# In[2]:


def create_dir(path): #crea carpetas si no existen 
    if not os.path.exists(path):
        os.makedirs(path)


# In[3]:


# from shutil import rmtree
# rmtree("DatasetPI2/Dosis/.ipynb_checkpoints")
# rmtree("DatasetPI2/Estructuras/.ipynb_checkpoints")


# ## Carga de datos

# In[4]:


def cargar_carpetas(carpeta, carpeta_estruc, carpeta_dosis ):
    direc = Path(carpeta +"/")
    estruc = [direc/carpeta_estruc/i for i in os.listdir(direc/carpeta_estruc)]
    dosis = [direc/carpeta_dosis/i for i in os.listdir(direc/carpeta_dosis)]
    estruc = sorted(estruc)
    dosis = sorted(dosis)
    print("Carpetas cargadas")
    if len(estruc) != len(dosis):
        print("La cantidad de volumenes no es equivalente a la cantidad de mascaras")
    print('La carpeta {} tiene {} volumenes y la carpeta {} tiene {} mascaras'.format(carpeta_estruc, len(estruc),carpeta_dosis, len(dosis)))
    
    return estruc, dosis


# # * Preprocesamiento

# In[5]:


def prepro (path): 
    lista_nii=[]
    carpeta = path.split("/")[-2].split(".")[0] #split da una lista con cada parte de la dirección separadas con un / y
    #el -2 el nombre de la imagen particular con la extensión y 
    #el segundo split da una lista con el nombre y la extención, con el [0] selecciona el nombre y desecha la extención
    nombre = carpeta[0:10].lower() #Si es Estructuras -> estructura, si es Dosis -> dosis
    for x in sorted(glob.iglob(f'{path}/*.nii')): # busca todos los punto nii en la carpeta 
        lista_nii.append(x) # agragar 
    for i,f in sorted(enumerate(lista_nii)): #numerate da el indice y direccion de cada item
        print (i)
        print (f)
        vol = nib.load(f).get_fdata()
        voltensor = torch.from_numpy(vol)
        voltensor = voltensor[None, None, :] 
        y = interpolate(voltensor, size = (64,64,64))
        input = y.numpy()
        inputnpy = np.resize(input,(64, 64, 64))
        create_dir ("DatasetPI2/"+ carpeta +"/")
        np.save(f'DatasetPI2/{carpeta}/{nombre}{i}',inputnpy)


# In[6]:


prepro ("DatasetPI2/Estructuras.nii/")


# In[7]:


prepro ("DatasetPI2/Dosis.nii/")


# # * Data augmentation

# ## Espejado de las estructuras y dosis

# In[8]:


def flip(path):
    lista_np = []
    carpeta = path.split("/")[-2].split(".")[0] 
    for x in sorted(glob.iglob(f'{path}/*.npy')):  
        lista_np.append(x) 
    for i,f in sorted(enumerate(lista_np)):
        print (i)
        nombre = f.split("/")[-1].split(".")[0] 
        print (f'flip_{nombre}')
        est = np.load(f)
        est_flip = np.flipud(est)
        create_dir ("DatasetPI/"+ carpeta +"_flip/")
        np.save(f'DatasetPI/{carpeta}_flip/flip_{nombre}',est_flip)


# In[9]:


flip("DatasetPI2/Estructuras/")


# In[10]:


flip("DatasetPI2/Dosis/")


# # Verificación del correcto funcionamiento

# In[5]:


estruc, dosis = cargar_carpetas('DatasetPI2', 'Estructuras', 'Dosis')
estruc_flip, dosis_flip = cargar_carpetas('DatasetPI2', 'Estructuras_flip', 'Dosis_flip')


# In[6]:


est = np.load('DatasetPI2/Estructuras/estructura6.npy')
dos = np.load('DatasetPI2/Dosis/dosis6.npy')
est_flip = np.load('DatasetPI2/Estructuras_flip/flip_estructura6.npy')
dos_flip = np.load('DatasetPI2/Dosis_flip/flip_dosis6.npy')


# In[7]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30,10))
i=32#random.randint(0, len(est)-1)
ax1.imshow(est[:,:,i],cmap="magma")
ax1.set_title("original", fontsize=30)
ax1.set_ylabel('x - Corte Transversal', fontsize=30)
ax1.set_xlabel('y', fontsize=30)
ax2.imshow(est_flip[:,:,i],cmap="magma")
ax2.set_title("epejada", fontsize=30)
ax2.set_ylabel('x', fontsize=30)
ax2.set_xlabel('y', fontsize=30)
ax3.imshow(est[:,:,i],cmap="magma")
ax3.imshow(dos[:,:,i], alpha=0.8,cmap="magma")
ax3.set_title("original con dosis", fontsize=30)
ax3.set_ylabel('x', fontsize=30)
ax3.set_xlabel('y', fontsize=30)
ax4.imshow(est_flip[:,:,i],cmap="magma")
ax4.imshow(dos_flip[:,:,i], alpha=0.8,cmap="magma")
ax4.set_title("epejada con dosis", fontsize=30)
ax4.set_ylabel('x', fontsize=30)
ax4.set_xlabel('y', fontsize=30)
plt.show()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30,10))
i=32#random.randint(0, len(est)-1)
ax1.imshow(est[:,i,:],cmap="magma")
ax1.set_ylabel('x - Corte Frontal', fontsize=30)
ax1.set_xlabel('z', fontsize=30)
ax2.imshow(est_flip[:,i,:],cmap="magma")
ax2.set_ylabel('x', fontsize=30)
ax2.set_xlabel('z', fontsize=30)
ax3.imshow(est[:,i,:],cmap="magma")
ax3.set_ylabel('x', fontsize=30)
ax3.set_xlabel('z', fontsize=30)
ax3.imshow(dos[:,i,:], alpha=0.8,cmap="magma")
ax4.imshow(est_flip[:,i,:],cmap="magma")
ax4.imshow(dos_flip[:,i,:], alpha=0.8,cmap="magma")
ax4.set_ylabel('x', fontsize=30)
ax4.set_xlabel('z', fontsize=30)
plt.show()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30,10))
i=32#random.randint(0, len(est)-1)
ax1.imshow(est[i,:,:],cmap="magma")
ax1.set_ylabel('y - Corte Sagital', fontsize=30)
ax1.set_xlabel('z', fontsize=30)
ax2.imshow(est_flip[i,:,:],cmap="magma")
ax2.set_ylabel('y', fontsize=30)
ax2.set_xlabel('z', fontsize=30)
ax3.imshow(est[i,:,:],cmap="magma")
ax3.imshow(dos[i,:,:], alpha=0.8,cmap="magma")
ax3.set_ylabel('y', fontsize=30)
ax3.set_xlabel('z', fontsize=30)
ax4.imshow(est_flip[i,:,:],cmap="magma")
ax4.imshow(dos_flip[i,:,:], alpha=0.8,cmap="magma")
ax4.set_ylabel('y', fontsize=30)
ax4.set_xlabel('z', fontsize=30)
plt.show()


# ## Grafico 3D

# In[8]:


import plotly.offline as pyo
pyo.init_notebook_mode()
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[9]:


fig = make_subplots(cols=2, specs=[[{'type': 'volume'}, {'type': 'volume'}]])
    
X, Y, Z = np.mgrid[-70:70:64j, -70:70:64j,-70:70:64j]
values1 = est
values2= est_flip

fig.add_trace(go.Volume(value=values1.flatten()), row=1, col=1)

fig.add_trace(go.Volume(value=values2.flatten()), row=1, col=2)

fig.update_traces(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), isomin=0.15, isomax=1, opacity=0.3, surface_count=10, colorscale='RdPu')
fig.show()


# In[10]:


data = np.load('DatasetPI2/Estructuras/estructura8.npy')
data_flip = np.load('DatasetPI2/Estructuras_flip/flip_estructura8.npy')


# In[11]:


fig = make_subplots(cols=2, specs=[[{'type': 'volume'}, {'type': 'volume'}]])
    
X, Y, Z = np.mgrid[-70:70:64j, -70:70:64j,-70:70:64j]
values1 = data
values2= data_flip

fig.add_trace(go.Volume(value=values1.flatten()), row=1, col=1)

fig.add_trace(go.Volume(value=values2.flatten()), row=1, col=2)

fig.update_traces(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), isomin=0.15, isomax=1, opacity=0.3, surface_count=10, colorscale='RdPu')
fig.show()


# # Dataset final

# In[26]:


# flip("DatasetPI/Estructuras/")


# In[27]:


# flip("DatasetPI/Dosis/")


# In[28]:


estruc, dosis = cargar_carpetas('DatasetPI', 'Estructuras', 'Dosis')
estruc_flip, dosis_flip = cargar_carpetas('DatasetPI', 'Estructuras_flip', 'Dosis_flip')


# In[29]:


# import shutil

# create_dir ("DatasetPI/EntrenamientoPI/Estructuras")
# create_dir ("DatasetPI/EntrenamientoPI/Dosis")

# for i in range(len(estruc)):
#     shutil.copy(estruc[i] , "DatasetPI/EntrenamientoPI/Estructuras")
#     shutil.copy(estruc_flip[i] , "DatasetPI/EntrenamientoPI/Estructuras")
#     shutil.copy(dosis[i] , "DatasetPI/EntrenamientoPI/Dosis")
#     shutil.copy(dosis_flip[i] , "DatasetPI/EntrenamientoPI/Dosis")


# In[30]:


estruc, dosis = cargar_carpetas('DatasetPI/EntrenamientoPI', 'Estructuras', 'Dosis')


# In[103]:


i=random.randint(0, (len(estruc)/2)-1)
est = np.load(estruc[i]) 
dos = np.load(dosis[i])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))
j=random.randint(0, len(est)-1)
ax1.imshow(est[:,:,j],cmap="magma")
ax2.imshow(dos[:,:,j],cmap="magma")
ax3.imshow(est[:,:,j],cmap="magma")
ax3.imshow(dos[:,:,j], alpha=0.8,cmap="magma")
plt.show()

est = np.load(estruc[i+120]) 
dos = np.load(dosis[i+120])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))
ax1.imshow(est[:,:,j],cmap="magma")
ax2.imshow(dos[:,:,j],cmap="magma")
ax3.imshow(est[:,:,j],cmap="magma")
ax3.imshow(dos[:,:,j], alpha=0.8,cmap="magma")
plt.show()


# In[31]:


estruc, dosis = cargar_carpetas('DatasetPI2', 'Estructuras', 'Dosis')
estruc_flip, dosis_flip = cargar_carpetas('DatasetPI2', 'Estructuras_flip', 'Dosis_flip')


# In[32]:


# import shutil

# create_dir ("DatasetPI/TesteoPI/Estructuras")
# create_dir ("DatasetPI/TesteoPI/Dosis")

# for i in range(len(estruc)):
#     shutil.copy(estruc[i] , "DatasetPI/TesteoPI/Estructuras")
#     shutil.copy(estruc_flip[i] , "DatasetPI/TesteoPI/Estructuras")
#     shutil.copy(dosis[i] , "DatasetPI/TesteoPI/Dosis")
#     shutil.copy(dosis_flip[i] , "DatasetPI/TesteoPI/Dosis")


# In[33]:


estruc, dosis = cargar_carpetas('DatasetPI/TesteoPI', 'Estructuras', 'Dosis')

