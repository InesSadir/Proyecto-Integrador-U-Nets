#!/usr/bin/env python
# coding: utf-8

# # U-Net 
# Mejoras: ReflectionPad3d (en vez de zero padding), posibilidad de agregar dropout, posibilidad de optar por upsamplig o convtranspose
# 

# In[1]:


import os 
import time
import math
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, jaccard_score


# In[2]:


"""
______________________________________encoding block_______________________________
"""
class encoding_block(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, padding=0, stride=1, dilation=1, dropout=True):
        super().__init__()
            
        layers = [nn.ReflectionPad3d(padding=(kernel_size -1)//2),
                  nn.Conv3d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                  nn.BatchNorm3d(out_size),
                  nn.PReLU(),
                  nn.ReflectionPad3d(padding=(kernel_size - 1)//2),
                  nn.Conv3d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                  nn.PReLU(),
                  nn.BatchNorm3d(out_size),
                  nn.PReLU(),
                  ]

        if dropout:
            layers.append(nn.Dropout())

        self.encoding_block = nn.Sequential(*layers)

    def forward(self, input):

        output = self.encoding_block(input)

        return output
"""
______________________________________decoding block_______________________________
"""
class decoding_block(nn.Module):
    def __init__(self, in_size, out_size, upsampling=True):
        super().__init__()

        if upsampling:
            self.up = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                    nn.Conv3d(in_size, out_size, kernel_size=1))

        else:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = encoding_block(in_size, out_size)

    def forward(self, input):

        output = self.up(input)

        return output

"""
______________________________________Main UNet architecture_______________________________
"""
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features=[64, 128, 256, 512, 1024], dropout=False,  upsampling=True):
        super().__init__()
        
        self.ups = nn.ModuleList() #no se puede usar una lista tipo [] porque se nececita almacenar capas convolucionales
        self.downs = nn.ModuleList() 
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
#Encoder
        for feature in features:
            self.downs.append(encoding_block(in_channels, feature, dropout = dropout))
            in_channels = feature
#Base        
        self.bottleneck = encoding_block(features[-1], features[-1]*2) #512 a 1024
            
#Decoder
        for feature in reversed(features):#los tamaños ahora van del ultimo al primero
            self.ups.append(decoding_block(feature*2, feature))
            self.ups.append(encoding_block(feature*2, feature))
            
#Final
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
                   

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]# da vuelta la lista para usarla en la subida
        
        for idx in range(0, len(self.ups), 2):#se hace un paso de 2 porque un solo paso va a equivaler al up and doubleconv
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

            
        return self.final_conv(x)


# ## * Comprobar funcionamiento 
# Se le ingresa a la arquitectura un tensor random con las dimensiones de los volúmenes de entrada y se verifica que la salida sea con las dimensiones de las dosis

# In[3]:


model = UNet()
x = torch.randn((1, 4, 64, 64, 64))
output = model(x)
output.shape


# # Métricas

# ## IoU

# In[4]:


def dice(true, pred, smooth=0.0000001):
    true=torch.round(true, decimals=1)
    pred=torch.round(pred, decimals=1)
    intersect = torch.sum((true == pred).int())
    union = math.prod(true.shape) + math.prod(pred.shape)
    return   torch.mean((2 * intersect) /(union+smooth))


# ## Dice

# In[5]:


def iou(true, pred, smooth=0.0000001):
    true=torch.round(true, decimals=1)
    pred=torch.round(pred, decimals=1)
    intersect = torch.sum((true == pred).int())
    union = math.prod(true.shape) + math.prod(pred.shape)-intersect
    return   torch.mean(intersect /(union+smooth))


# # BCEDiceLoss

# In[6]:


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, true, pred):
        true = true.view(-1)
        pred = pred.view(-1)
        # BCE loss
        bce_loss = nn.BCEWithLogitsLoss()(true, pred).double()
        # Dice Loss
        dice_coef = ((-(true-pred)**2+1).double().sum()) / (math.prod(true.shape))
        return bce_loss + (1 - dice_coef)


# # Utils

# ## Seed

# In[7]:


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    


# ## Calculo del tiempo

# In[8]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins*60))
    
    return elapsed_mins, elapsed_secs


# ## Data loader

# In[9]:


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# In[10]:


class Dataset (torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
       
    def __len__(self):
        assert len(self.x) == len(self.y), "La cantidad de volumenes no es equivalente a la cantidad de mascaras"
        return len(self.x)
    
    
    def __getitem__(self,ix):
        img = np.load(self.x[ix])
        mask = np.load(self.y[ix])
        img_oh = (np.arange(4) == img[...,None]).astype(np.float64)
        img_tensor= torch.tensor(img_oh).permute(3, 0, 1, 2)
        mask_e = mask/36.25
        mask_tensor = torch.tensor(mask_e).unsqueeze(0)
        
        return img_tensor, mask_tensor


# In[11]:


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


# # * Entrenamiento de un solo volumen

# In[12]:


estruc, dosis = cargar_carpetas('DatasetPI/EntrenamientoPI', 'Estructuras', 'Dosis')
dataset = Dataset(estruc[63:64], dosis[63:64])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
x, y = next(iter(dataloader))


# In[13]:


device = torch.device('cuda')
model = UNet()
model = model.to(device)


# In[14]:


def fit(model, X, y, epochs=1, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.to(device)
    X = X.to(device=device, dtype=torch.float)
    y = y.to(device=device, dtype=torch.float)
    model.train()
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        ious = iou(y_hat, y)
        dices = dice(y_hat, y)
        print(f"Epoch {epoch}/{epochs} loss {loss.item():.5f} iou {ious:.3f} dice {dices:.3f}")


# In[12]:


fit(model, x, y, epochs=5000)


# In[17]:


model.eval()
with torch.no_grad():
    output = model(x.to(device=device, dtype=torch.float))[0]


# In[18]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))
ax1.imshow(y[0,0,:,:,24],cmap="magma")
ax2.imshow(output[0,:,:,24].squeeze().cpu().numpy(),cmap="magma")
plt.show()


# # * Resumen de arquitectura

# In[19]:


summary(model, (4, 64, 64, 64))


# # Entrenamiento

# In[13]:


estruc, dosis = cargar_carpetas('DatasetPI/EntrenamientoPI', 'Estructuras', 'Dosis')


# In[14]:


dataset = {
    "train" : Dataset(estruc[:180], dosis[:180]),
    "test" : Dataset(estruc[180:], dosis[180:])}

print(f'La cantidad de imágenes de entrenamiento son {len(dataset["train"])} y la cantidad de validación son {len(dataset["test"])}.')


# In[15]:


dataloader = {
    "train" : torch.utils.data.DataLoader(dataset["train"], batch_size=2, shuffle=True, pin_memory=True),
    "test" : torch.utils.data.DataLoader(dataset["test"], batch_size=2, pin_memory=True) }

imges, maskes = next(iter(dataloader["train"]))
imges.shape, maskes.shape


# In[16]:


def train(model, dataloader, hist, met, optimizer, loss_fn, device):
    epoch_loss = 0.0
    bar = tqdm(dataloader['train'])
    train_loss= []
    model.train()
    for imges, maskes in bar:
        imges, maskes = imges.to(device, dtype=torch.float), maskes.to(device, dtype=torch.float)
        optimizer.zero_grad()
        y_pred = model(imges)
        loss = loss_fn(y_pred, maskes)
        loss.backward() 
        optimizer.step() 
        ious = iou(y_pred, maskes)
        dices = dice(y_pred, maskes)
        train_loss.append(loss.item())
        epoch_loss += loss.item()
        bar.set_description(f"loss {np.mean(train_loss):.5f} iou {ious:.3f} dice {dices:.3f}")
    hist['loss'].append(np.mean(train_loss))
    met["IoU"].append(np.mean(ious.cpu().numpy()*1))
    met["Dice"].append(np.mean(dices.cpu().numpy()*1))
    epoch_loss = epoch_loss/len(bar)
    
    return epoch_loss, hist, met


# In[17]:


def evaluate(model, dataloader, hist, met, loss_fn, device):
    epoch_loss = 0.0
    bar = tqdm(dataloader['test'])
    test_loss = []
    model.eval()
    with torch.no_grad():
        for imges, maskes in bar:
            imges, maskes = imges.to(device,dtype=torch.float), maskes.to(device, dtype=torch.float)
            y_pred = model(imges)
            loss = loss_fn(y_pred, maskes)
            epoch_loss += loss.item()
            test_loss.append(loss.item())
            ious = iou(y_pred, maskes)
            dices = dice(y_pred, maskes)
            bar.set_description(f"test_loss {np.mean(test_loss):.5f} iou {ious:.3f} dice {dices:.3f}")
        hist['test_loss'].append(np.mean(test_loss))
        met["IoU_test"].append(np.mean(ious.cpu().numpy()*1))
        met["Dice_test"].append(np.mean(dices.cpu().numpy()*1))
        epoch_loss = epoch_loss/len(bar)
    
    return epoch_loss, hist, met


# In[19]:


device = torch.device('cuda')
model = UNet()
model = model.to(device)


# In[20]:


def fit(epochs=10, loss_fn = torch.nn.MSELoss(), model=model, device=device):
    if __name__ == "__main__":

        seeding(42)
        create_dir("file")

        epochs = epochs
        lr = 1e-4
        checkpoint_path = "file/checkpoint.pth" # Para guardar el modelo 

        dataloaders = dataloader

        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr= 1e-8)
        loss_fn = loss_fn
        
        # Entrenamiento
        hist = {'loss': [], 'test_loss': []}
        met = {"IoU": [], "Dice": [], "IoU_test": [], "Dice_test": []}
        best_valid_loss = float("inf")
        start_time = time.time()

        for epoch in range(1, epochs+1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, hist, met = train(model, dataloaders, hist, met, optimizer, loss_fn, device)
            valid_loss, hist, met = evaluate(model, dataloaders, hist, met, loss_fn, device)
        # Guardar el modelo entrenado 
            if valid_loss < best_valid_loss:
                data_str = f"Valid loss improved from {best_valid_loss:.4f} to {valid_loss:.4f}. Saving checkpoint: {checkpoint_path}"
                print(data_str)

                best_valid_loss = valid_loss
                torch.save(model.state_dict(), checkpoint_path)
        # Mostrar avance del entrenamiento    
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        data_str = f' Entrenamiento de {epochs} epocas finalizado en {epoch_mins}m {epoch_secs}s\n'
        print()
        print(data_str)
        
        return hist, met


# ## * Entrenamiento de 10 épocas con torch.nn.MSELoss
# 

# In[19]:


hist, met = fit(epochs=10, loss_fn = torch.nn.MSELoss())


# In[20]:


#10 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[21]:


#10 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[22]:


#10 epocas
model.eval()
for i in range(5):
    with torch.no_grad():
        ix = random.randint(0, len(dataset['test'])-1)
        img, mask = dataset['test'][ix]
        output = model(img.unsqueeze(0).to(device, dtype=torch.float))[0]
        pred_mask = output
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.set_title('Objetivo', fontsize=30)
    ax1.imshow(mask[:,:,:,29].permute(1,2,0),cmap="magma")
    ax2.set_title('Predicción', fontsize=30)
    ax2.imshow(pred_mask.squeeze().cpu().numpy()[:,:,29],cmap="magma")
    plt.show()


# ## * Entrenamiento de 100 épocas con torch.nn.MSELoss
# 

# In[23]:


hist, met = fit(epochs=100, loss_fn = torch.nn.MSELoss())


# In[24]:


#100 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[26]:


#100 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[27]:


#100 epocas
model.eval()
for i in range(5):
    with torch.no_grad():
        ix = random.randint(0, len(dataset['test'])-1)
        img, mask = dataset['test'][ix]
        output = model(img.unsqueeze(0).to(device, dtype=torch.float))[0]
        pred_mask = output
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.set_title('Objetivo', fontsize=30)
    ax1.imshow(mask[:,:,:,29].permute(1,2,0),cmap="magma")
    ax2.set_title('Predicción', fontsize=30)
    ax2.imshow(pred_mask.squeeze().cpu().numpy()[:,:,29],cmap="magma")
    plt.show()


# ## * Entrenamiento de 10 épocas con torch.nn.L1Loss
# 

# In[25]:


hist, met = fit(epochs=10, loss_fn = torch.nn.L1Loss())


# In[26]:


#10 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[27]:


#10 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[28]:


#10 epocas
model.eval()
for i in range(5):
    with torch.no_grad():
        ix = random.randint(0, len(dataset['test'])-1)
        img, mask = dataset['test'][ix]
        output = model(img.unsqueeze(0).to(device, dtype=torch.float))[0]
        pred_mask = output
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.set_title('Objetivo', fontsize=30)
    ax1.imshow(mask[:,:,:,29].permute(1,2,0),cmap="magma")
    ax2.set_title('Predicción', fontsize=30)
    ax2.imshow(pred_mask.squeeze().cpu().numpy()[:,:,29],cmap="magma")
    plt.show()


# ## * Entrenamiento de 100 épocas con torch.nn.L1Loss
# 

# In[19]:


hist, met = fit(epochs=100, loss_fn = torch.nn.L1Loss())


# In[20]:


#100 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[21]:


#100 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[22]:


#100 epocas
model.eval()
for i in range(5):
    with torch.no_grad():
        ix = random.randint(0, len(dataset['test'])-1)
        img, mask = dataset['test'][ix]
        output = model(img.unsqueeze(0).to(device, dtype=torch.float))[0]
        pred_mask = output
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.set_title('Objetivo', fontsize=30)
    ax1.imshow(mask[:,:,:,29].permute(1,2,0),cmap="magma")
    ax2.set_title('Predicción', fontsize=30)
    ax2.imshow(pred_mask.squeeze().cpu().numpy()[:,:,29],cmap="magma")
    plt.show()

