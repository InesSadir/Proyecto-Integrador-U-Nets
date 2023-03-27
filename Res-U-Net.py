#!/usr/bin/env python
# coding: utf-8

# # Res-U-Net 
# Mediante el método de transfer learning se hace uso de la red pre entrenada ResNet18.
# 

# In[1]:


import os 
import time
import math
import torch
import random
import torchvision
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


# ## ResNet versión 2D

# In[2]:


resnet=torchvision.models.resnet18()
resnet


# ## ResNet versión 3D

# In[3]:


resnet=torchvision.models.video.r3d_18()
resnet


# # Adaptación de la primera capa para los datos de entrada propios

# In[4]:


encoder = torchvision.models.video.r3d_18(weights="R3D_18_Weights.DEFAULT")           
encoder.stem = torch.nn.Sequential(
    nn.Conv3d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),# Se debe cambiar la cantidad de canales de entrada a la cantidad dada por el one-hot encoding
    nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(inplace=True))
encoder


# # Implementación 

# In[2]:


def conv3x3(in_size, out_size):
    return torch.nn.Sequential(
        nn.ReflectionPad3d(padding=(3 -1)//2),
        nn.Conv3d(in_size, out_size, 3, padding=0),
        nn.BatchNorm3d(out_size),
        nn.PReLU(),
        nn.Dropout())

class deconv(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(deconv, self).__init__()
        self.upsample = torch.nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.conv1 = conv3x3(in_size, out_size)
        self.conv2 = conv3x3(out_size, out_size)
    
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class out_conv(torch.nn.Module):
    def __init__(self, in_size, out_size, final_size):
        super(out_conv, self).__init__()
        self.upsample = torch.nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.conv = conv3x3(in_size, out_size)
        self.final = torch.nn.Conv3d(out_size, final_size, 1)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = self.conv(x1) 
        x = self.final(x)
        return x

class ResUNet(torch.nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super().__init__()
          
        self.encoder = torchvision.models.video.r3d_18(weights="R3D_18_Weights.DEFAULT")           
        self.encoder.stem = torch.nn.Sequential(
            nn.Conv3d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.deconv1 = deconv(512,256)
        self.deconv2 = deconv(256,128)
        self.deconv3 = deconv(128,64)
        self.out = out_conv(64, 64, out_ch)

    def forward(self, x):
        x_in = x.clone().detach()
        #print(x_in.shape)
        x = self.encoder.stem(x)
        #print(x.shape)
        x1 = self.encoder.layer1(x)
        #print(x1.shape)
        x2 = self.encoder.layer2(x1)
        #print(x2.shape)
        x3 = self.encoder.layer3(x2)
        #print(x3.shape)
        x = self.encoder.layer4(x3)
        #print(x.shape)
        x = self.deconv1(x, x3)
        #print(x.shape)
        x = self.deconv2(x, x2)
        #print(x.shape)
        x = self.deconv3(x, x1)
        #print(x.shape)
        x = self.out(x, x_in)
        #print(x.shape)
        return x


# ## * Comprobar funcionamiento

# In[6]:


model = ResUNet()
x = torch.randn((1, 4, 64, 64, 64))
output = model(x)
output.shape


# # Métricas

# ## Dice 

# In[3]:


def dice(true, pred, smooth=0.0000001):
    true=torch.round(true, decimals=1)
    pred=torch.round(pred, decimals=1)
    intersect = torch.sum((true == pred).int())
    union = math.prod(true.shape) + math.prod(pred.shape)
    return   torch.mean((2 * intersect) /(union+smooth))


# ## IoU

# In[4]:


def iou(true, pred, smooth=0.0000001):
    true=torch.round(true, decimals=1)
    pred=torch.round(pred, decimals=1)
    intersect = torch.sum((true == pred).int())
    union = math.prod(true.shape) + math.prod(pred.shape)-intersect
    return   torch.mean(intersect /(union+smooth))


# # BCEDiceLoss
# 

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

# In[16]:


estruc, dosis = cargar_carpetas('DatasetPI/EntrenamientoPI', 'Estructuras', 'Dosis')
dataset = Dataset(estruc[63:64], dosis[63:64])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


# In[17]:


x, y = next(iter(dataloader))


# In[18]:


device = torch.device('cuda')
model = ResUNet()
model = model.to(device)


# In[22]:


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


# In[1]:


fit(model, x, y, epochs=5000)


# In[24]:


model.eval()
with torch.no_grad():
    output = model(x.to(device=device, dtype=torch.float))[0]


# In[25]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))
ax1.imshow(y[0,0,:,:,24],cmap="magma")
ax2.imshow(output[0,:,:,24].squeeze().cpu().numpy(),cmap="magma")
plt.show()


# # * Resumen de arquitectura

# In[26]:


summary(model, (4, 64, 64, 64))


# # Entrenamiento

# In[12]:


estruc, dosis = cargar_carpetas('DatasetPI/EntrenamientoPI', 'Estructuras', 'Dosis')


# In[13]:


dataset = {
    "train" : Dataset(estruc[:180], dosis[:180]),
    "test" : Dataset(estruc[180:], dosis[180:])}

print(f'La cantidad de imágenes de entrenamiento son {len(dataset["train"])} y la cantidad de validación son {len(dataset["test"])}.')


# In[14]:


dataloader = {
    "train" : torch.utils.data.DataLoader(dataset["train"], batch_size=2, shuffle=True, pin_memory=True),
    "test" : torch.utils.data.DataLoader(dataset["test"], batch_size=2, pin_memory=True) }

imges, maskes = next(iter(dataloader["train"]))
imges.shape, maskes.shape


# In[15]:


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


# In[16]:


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


# In[17]:


device = torch.device('cuda')
model = ResUNet()
model = model.to(device)


# In[18]:


def fit(epochs=10, loss_fn = torch.nn.MSELoss(), model=model):
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

# In[18]:


hist, met = fit(epochs=10, loss_fn = torch.nn.MSELoss())


# In[19]:


#10 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[20]:


#10 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[21]:


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

# In[22]:


hist, met = fit(epochs=100, loss_fn = torch.nn.MSELoss())


# In[23]:


#100 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[24]:


#100 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[25]:


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


# ## * Entrenamiento de 10 épocas con BCEWithLogitsLoss

# In[27]:


hist, met = fit(epochs=10, loss_fn = torch.nn.BCEWithLogitsLoss())


# In[28]:


#10 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[29]:


#10 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[30]:


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


# ## * Entrenamiento de 10 épocas con torch.nn.L1Loss

# In[22]:


hist, met = fit(epochs=10, loss_fn = torch.nn.L1Loss())


# In[23]:


#10 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[24]:


#10 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[26]:


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


# In[23]:


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


# ## * Entrenamiento de 10 épocas con BCEDiceLoss

# In[34]:


hist, met = fit(epochs=10, loss_fn = BCEDiceLoss())


# In[35]:


#10 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[36]:


#10 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[37]:


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


# ## * Entrenamiento de 100 épocas con BCEDiceLoss

# In[40]:


hist, met = fit(epochs=100, loss_fn = BCEDiceLoss())


# In[41]:


#100 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[42]:


#100 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[43]:


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

