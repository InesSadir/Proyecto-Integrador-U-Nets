#!/usr/bin/env python
# coding: utf-8

# # Carga de datos 
# El dataset se cargó a la notebook por medio del terminal de ubuntu con el siguiente código:
# 
# ls: MUESTRA LAS CARPETAS DISPONIBLES
# 
# cd: PARA UBICARSE EN UNA CARPETA
# 
# scp -r </home/ines/pps/DatasetPI> isadir@nabucodonosor.ccad.unc.edu.ar:~/ : CARGA LA CARPETA DatasetPI
# 
# cd~: PARA SALIR DE LAS CARPETAS
# 
# Se genera una función que recibe la tres carpetas, en este caso en la carpeta DatasetPI se tiene la carpeta Estructuras con las imágenes de las estructuras demarcadas (que son las x) y la carpeta Dosis con las dosis (que son las y) correspondientes. Además se verifica que existan la misma cantidad de volúmenes y dosis.

# In[1]:


import os 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def cargar_carpetas(carpeta, carpeta_estruc, carpeta_dosis ):
    direc = Path(carpeta +"/")
    estruc = [direc/carpeta_estruc/i for i in os.listdir(direc/carpeta_estruc)]
    dosis = [direc/carpeta_dosis/i for i in os.listdir(direc/carpeta_dosis)]
    estruc = sorted(estruc)
    dosis = sorted(dosis)
    print("Carpetas cargadas")
    if len(estruc) != len(dosis):
        print("La cantidad de volúmenes no es equivalente a la cantidad de máscaras")
    print('La carpeta {} tiene {} volúmenes y la carpeta {} tiene {} máscaras'.format(carpeta_estruc, len(estruc),carpeta_dosis, len(dosis)))
    
    return estruc, dosis


# # * Visualización de los datos
# A continuación, se muestra un ejemplo de superposición de unas estructuras con su correspondiente distribución de dosis planificada y se examina el tipo de dato.

# In[3]:


estruc, dosis = cargar_carpetas('DatasetPI', 'Estructuras', 'Dosis')


# In[4]:


# from shutil import rmtree

# rmtree("DatasetPI/Dosis/.ipynb_checkpoints")
# rmtree("DatasetPI/Estructuras/.ipynb_checkpoints")


# In[5]:


plt.figure(figsize=(40,40))
for i in range(70):
    plt.subplot(10, 7, i+1)
    img = np.load(dosis[i])
    mask = np.load(estruc[i])
    plt.title(i, size=40)
    plt.imshow(img[:,:,23],cmap="magma")
    plt.imshow(mask[:,:,23], alpha=0.4,cmap="magma")
    plt.axis('Off')    
plt.tight_layout()
plt.show() 


# In[6]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
est = np.load(estruc[63]) 
dos = np.load(dosis[63])
ax1.imshow(est[:,:,23],cmap="magma")
ax1.set_title("Estructuras --> input",size=30 )
ax2.imshow(dos[:,:,23],cmap="magma")
ax2.set_title("Dosis --> target",size=30 )
im=ax2.imshow(dos[:,:,23],cmap="magma")
ax3.imshow(est[:,:,23],cmap="magma")
ax3.imshow(dos[:,:,23], alpha=0.8,cmap="magma")
ax3.set_title("Superposición dosis-PTV",size=30 )
cbar = plt.colorbar(im, orientation="horizontal", ax = [ax2, ax3], pad=0.05, aspect=100)
cbar.ax.tick_params(labelsize=20)
plt.show()


# El tipo de dato de ambos volúmenes es float64 aunque técnicamente los volúmenes de estructuras son enteros ya que se representa con valor 0, 1, 2 y 3 a los pixeles del fondo, recto próstata (PTV) y vejiga . Por el lado de las dosis el valor de prescripción es de 36,25 Gy en el PTV pero por cuestiones geométricas y de tecnología hay puntos donde la dosis puede llegar a valores alrededor de 40.

# In[6]:


type(est), est.shape, est.dtype, est.min(), est.max()


# In[7]:


type(dos), dos.shape, dos.dtype, dos.min(), dos.max()


# # * Preparación de los datos
# A las imágenes de las estructuras se pasan a formato One Hot Encoding que genera para cada pixel un vector con la cantidad de posiciones según la cantidad de estructuras que se tenga y le asigna el valor 1 a la posición que corresponda a la estructura que representa dicho pixel y 0 a todas las demás. A las máscaras de las dosis se las escalar en valores aproximadamente entre 1 y 0 considerando que el valor máximo es 36,25.

# In[8]:


est_oh = (np.arange(4) == est[...,None]).astype(np.float64) 
type(est_oh), est_oh.shape, est_oh.dtype, est_oh.max(), est_oh.min()


# In[9]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25,10))
ax1.imshow(est_oh[:,:,24,0],cmap="magma")
ax1.set_title("fondo",size=20 )
ax2.imshow(est_oh[:,:,24,1],cmap="magma")
ax2.set_title("recto",size=20 )
ax3.imshow(est_oh[:,:,24,3],cmap="magma")
ax3.set_title("próstata",size=20 )
ax4.imshow(est_oh[:,:,24,2],cmap="magma")
im = ax4.imshow(est_oh[:,:,24,2],cmap="magma")
ax4.set_title("vejiga",size=20 )
cbar = plt.colorbar(im, orientation="horizontal", ax = [ax1, ax2, ax3, ax4], pad=0.05, aspect=100)
cbar.ax.tick_params(labelsize=20)
plt.show()


# In[10]:


dos_e = dos/36.25
type(dos_e), dos_e.shape, dos_e.dtype, dos_e.max(), dos_e.min()


# ## Generalizando
# Con el modulo torch.utils.data de PyTorh se cargan los datos. La clase Dataset permiten instanciar objetos con el conjunto de datos que se van a cargar. Luego se aplica el preprocesamiento necesario, es decir el One Hot Encoding para las estructuras y el escalamiento para las dosis. 
# 

# In[3]:


import torch.utils.data


# In[4]:


class Dataset (torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
       
    def __len__(self):
        assert len(self.x) == len(self.y), "La cantidad de volúmenes no es equivalente a la cantidad de máscaras"
        return len(self.x)
    
    
    def __getitem__(self,ix):
        img = np.load(self.x[ix])
        mask = np.load(self.y[ix])
        img_oh = (np.arange(4) == img[...,None]).astype(np.float64)
        img_tensor= torch.tensor(img_oh).permute(3, 0, 1, 2)
        mask_e = mask/36.25
        mask_tensor = torch.tensor(mask_e).unsqueeze(0)
        
        return img_tensor, mask_tensor


# # U-Net

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[6]:


def conv3x3_bn(ci, co):
    return torch.nn.Sequential(
        torch.nn.Conv3d(ci, co, 3, padding=1),
        torch.nn.BatchNorm3d(co),
        torch.nn.ReLU(inplace=True)
    )

def encoder_conv(ci, co):
    return torch.nn.Sequential(
        torch.nn.MaxPool3d((2, 2, 2)),
        conv3x3_bn(ci, co),
        conv3x3_bn(co, co),
    )

class deconv(torch.nn.Module):
    def __init__(self, ci, co):
        super(deconv, self).__init__()
        self.upsample = torch.nn.ConvTranspose3d(ci, co, 2, stride=2, padding=0)
        self.conv1 = conv3x3_bn(ci, co)
        self.conv2 = conv3x3_bn(co, co)
    
    # recibe la salida de la capa anetrior y la salida de la etapa
    # correspondiente del encoder
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
#         diffX = x2.size()[2] - x1.size()[2] # si no se usa zero padding en las convoluciónes se deben cortar antes de cat
#         diffY = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, (diffX, 0, diffY, 0))
        # concatenamos los tensores
        x = torch.cat([x2, x1], axis=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self, n_classes=4, in_ch=4):
        super().__init__()

        # lista de capas en encoder-decoder con número de filtros
        c = [64, 128, 256, 512]

        # primera capa conv que recibe la imagen
        self.conv1 = torch.nn.Sequential(
          conv3x3_bn(in_ch, c[0]),
          conv3x3_bn(c[0], c[0]),
        )
        # capas del encoder
        self.conv2 = encoder_conv(c[0], c[1])
        self.conv3 = encoder_conv(c[1], c[2])
        self.conv4 = encoder_conv(c[2], c[3])

        # capas del decoder
        self.deconv1 = deconv(c[3],c[2])
        self.deconv2 = deconv(c[2],c[1])
        self.deconv3 = deconv(c[1],c[0])

        # útlima capa conv que nos da la máscara
        self.out = torch.nn.Conv3d(c[0],  1, kernel_size=1, padding=0)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = self.conv4(x3)
        # decoder
        x = self.deconv1(x, x3)
        x = self.deconv2(x, x2)
        x = self.deconv3(x, x1)
        x = self.out(x)
        return x


# ## * Comprobar funcionamiento 
# Se le ingresa a la arquitectura un tensor random con las dimensiones de los volúmenes de entrada y se verifica que la salida sea con las dimensiones de las dosis

# In[15]:


model = UNet()
output = model(torch.randn((1, 4, 64, 64, 64)))
output.shape


# In[16]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[17]:


dos_tensor = torch.tensor(dos_e).unsqueeze(0).unsqueeze(0)
est_tensor= torch.tensor(est_oh).permute(3, 0, 1, 2).unsqueeze(0)


# # Métricas 

# In[7]:


import math
from sklearn.metrics import f1_score, jaccard_score


# ## Dice
# Es una métrica que mide la similitud con la relación entre la intersección por dos y la suma del área total de las dos figuras. 

# In[8]:


def dice(true, pred, smooth=0.0000001):
    true=torch.round(true, decimals=1) # se redondea con un decimal los valores de los pixeles para que las coincidencias no sea tan excasas 
    pred=torch.round(pred, decimals=1) # como lo son cuando se tienen 6 valores despues de la coma.
    intersect = torch.sum((true == pred).int()) # la intersección son los pixeles en donde coincide el valor en los dos ejemplos 
    total = math.prod(true.shape) + math.prod(pred.shape) # el total es la suma de todos los pixeles que forman ambos ejemplos
    return   torch.mean((2 * intersect) /(total +smooth)) 


# ## IoU
# Intersection over Union se usa para evaluar, es la relación entre las predicciones correctas y el área total combinada de predicción y verdad del terreno. Como los targets y las predicciones no son imágenes estrictamente binarias se adaptó de tal forma que se contabilicen los pixeles que tengan el mismo valor (redondeado) y no únicamente los que se encuentren por arriba de un umbral. 

# In[9]:


def iou(true, pred, smooth=0.0000001):
    true=torch.round(true, decimals=1)
    pred=torch.round(pred, decimals=1)
    intersect = torch.sum((true == pred).int())
    union = math.prod(true.shape) + math.prod(pred.shape)-intersect
    return   torch.mean(intersect /(union+smooth))


# ## F1 y Jaccard
# Métricas del paquete de sklearn, F1 es matemáticamente equivalente a Dice y F1 a IoU.

# In[10]:


def pre (y):
    y = y.cpu().numpy()
    y = y.astype(np.uint8)
    y = y.reshape(-1)
    return y

def f1 (true, pred):
    y_true = pre(true)
    y_pred = pre(pred)
    return f1_score(y_true, y_pred, average='macro')

def jac (true, pred):
    y_true = pre(true)
    y_pred = pre(pred)
    return jaccard_score(y_pred,  y_true, average='macro') 


# # BCEDiceLoss

# In[11]:


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


# # Sharp Loss

# In[12]:


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, true, pred):
        true = true.view(-1)
        pred = pred.view(-1)
        factor=1/(1+torch.exp(-(true-0.03)*100))
        # Dice Loss
        dice_coef = ((-(true-pred)**2+1).double().sum()) / (math.prod(true.shape))
        return 1 - factor.sum()


# In[13]:


class SharpLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        
    def forward(self, true, pred):
        true = true.view(-1)
        pred = pred.view(-1)
        n=math.prod(true.shape)
        factor=1/(1+torch.exp(-(true-0.03)*100))
        dif=(true-pred)**2
        suma= (factor*dif).double().sum()
        final=suma/n
        return final


# # * Entrenamiento de un solo volumen
# Para verificar que no hay errores de código primero se corre un entrenamiento para un solo ejemplo.

# In[25]:


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


# In[7]:


fit(model, est_tensor, dos_tensor, epochs=5000)


# In[27]:


model.eval()
with torch.no_grad():
    output = model(est_tensor.to(device=device, dtype=torch.float))[0]


# In[28]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))
ax1.imshow(est[:,:,24],cmap="magma")
ax2.imshow(dos[:,:,24],cmap="magma")
ax3.imshow(output[:,:,24].squeeze().cpu().numpy(),cmap="magma")
plt.show()


# # * Summary 
# Descripción detallada de las capas del modelo y los parámetros.

# In[29]:


from torchsummary import summary
summary(model, (4, 64, 64, 64))


# # Calculo de tiempo

# In[14]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins*60))
    
    return elapsed_mins, elapsed_secs


# # Entrenamiento
# Primero se dividen los datos para entrenamiento, los 100 primeros y testeo, los 23 restantes. Luego se carga y preprocesan los datos.

# In[15]:


from tqdm import tqdm
import pandas as pd
import random
import time


# In[16]:


estruc, dosis = cargar_carpetas('DatasetPI/EntrenamientoPI', 'Estructuras', 'Dosis')


# In[17]:


dataset = {
    "train" : Dataset(estruc[:180], dosis[:180]),
    "test" : Dataset(estruc[180:], dosis[180:])}

len(dataset["train"]), len(dataset["test"])


# In[24]:


dataloader = {
    "train" : torch.utils.data.DataLoader(dataset["train"], batch_size=16, shuffle=True, pin_memory=True),
    "test" : torch.utils.data.DataLoader(dataset["test"], batch_size=16, pin_memory=True) }

imges, maskes = next(iter(dataloader["train"]))
imges.shape, maskes.shape


# In[25]:


def fit(model, dataloader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.to(device)
    hist = {'loss': [], 'test_loss': []}
    met = {'iou': [], 'dice': [], 'iou_test': [], 'dice_test': []}
    start_time = time.time()
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        bar = tqdm(dataloader['train'])
        train_loss= []
        model.train()
        for imges, maskes in bar:
            imges, maskes = imges.to(device, dtype=torch.float), maskes.to(device, dtype=torch.float)
            optimizer.zero_grad()
            y_hat = model(imges)
            loss = criterion(y_hat, maskes)
            loss.backward()
            optimizer.step()
            ious = iou(y_hat, maskes)
            dices = dice(y_hat, maskes)
            train_loss.append(loss.item())
            bar.set_description(f"loss {np.mean(train_loss):.5f} iou {ious:.3f} dice {dices:.3f}")
        hist['loss'].append(np.mean(train_loss))
        met['dice'].append(dices.cpu().numpy()*1)
        met['iou'].append(ious.cpu().numpy()*1)
        bar = tqdm(dataloader['test'])
        test_loss = []
        model.eval()
        with torch.no_grad():
            for imges, maskes in bar:
                imges, maskes = imges.to(device,dtype=torch.float), maskes.to(device, dtype=torch.float)
                y_hat = model(imges)
                loss = criterion(y_hat, maskes)
                test_loss.append(loss.item())
                ious = iou(y_hat, maskes)
                dices = dice(y_hat, maskes)
                bar.set_description(f"test_loss {np.mean(test_loss):.5f} iou {ious:.3f} dice {dices:.3f}")
        hist['test_loss'].append(np.mean(test_loss))
        met['dice_test'].append(dices.cpu().numpy()*1)
        met['iou_test'].append(ious.cpu().numpy()*1)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    data_str = f' Entrenamiento de {epochs} epocas finalizado en {epoch_mins}m {epoch_secs}s\n'
    print()
    print(data_str)
    
    return hist, met


# ## * Entrenamiento de 10 épocas con torch.nn.MSELoss

# In[22]:


device = torch.device('cuda')
model = UNet()
hist, met = fit(model, dataloader)


# In[49]:


#10 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[51]:


#10 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[52]:


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

# In[58]:


#En Nabucodonosor
model = UNet()
hist, met = fit(model, dataloader, epochs=100)


# In[26]:


#En Mendieta
model = UNet()
hist, met = fit(model, dataloader, epochs=100)


# In[29]:


#100 epocas en nabu
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(6,6))
plt.show()


# In[59]:


#100 epocas en mendieta
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[31]:


#100 epocas en nabu
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(5.5,5.5))
plt.show()


# In[60]:


#100 epocas en mendieta
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[34]:


model.eval()
for i in range(20):
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


# In[61]:


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


# ## * Entrenamiento de 500 épocas con torch.nn.MSELoss

# In[62]:


device = torch.device('cuda')
model = UNet()
hist, met = fit(model, dataloader, epochs=500)


# In[63]:


#500 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[64]:


#500 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[65]:


#500 epocas
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

# In[29]:


device = torch.device('cuda')
model = UNet()
hist, met = fit(model, dataloader, epochs=10)


# In[30]:


#10 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[31]:


#10 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[23]:


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


# ## * Entrenamiento de 100 épocas con BCEWithLogitsLoss

# In[25]:


device = torch.device('cuda')
model = UNet()
hist, met = fit(model, dataloader, epochs=100)


# In[26]:


#100 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[27]:


#100 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[28]:


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


# ## * Entrenamiento de 10 épocas con L1Loss

# In[33]:


device = torch.device('cuda')
model = UNet()
hist, met = fit(model, dataloader, epochs=10)


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


# ## * Entrenamiento de 100 épocas con L1Loss

# In[38]:


device = torch.device('cuda')
model = UNet()
hist, met = fit(model, dataloader, epochs=100)


# In[39]:


#100 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[40]:


#100 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[41]:


#100 epocas
for i in range(5):
    model.eval()
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


# ## * Entrenamiento de 500 épocas con L1Loss

# In[42]:


device = torch.device('cuda')
model = UNet()
hist, met = fit(model, dataloader, epochs=500)


# In[43]:


#500 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[44]:


#500 epocas
df = pd.DataFrame(met)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[45]:


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

# In[19]:


device = torch.device('cuda')
model = UNet()
hist, met = fit(model, dataloader, epochs=10)


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


# ## * Entrenamiento de 100 épocas con BCEDiceLoss

# In[23]:


device = torch.device('cuda')
model = UNet()
hist, met = fit(model, dataloader, epochs=100)


# In[24]:


#100 epocas
df = pd.DataFrame(hist)
df.plot(grid=True, figsize=(8,8))
plt.show()


# In[25]:


#100 epocas
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

