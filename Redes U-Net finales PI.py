#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import time
import math
import torch
import random
import statistics
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
from glob import glob
import torch.utils.data
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
from torchsummary import summary
import plotly.graph_objects as go
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
from torchvision.models.video import r3d_18, R3D_18_Weights


# # Carga de datos

# In[2]:


def cargar_carpetas(carpeta, carpeta_estruc, carpeta_dosis):
    direc = Path(carpeta +"/") # carpeta general
    estruc = [direc/carpeta_estruc/i for i in os.listdir(direc/carpeta_estruc)] # carpeta de estructuras
    dosis = [direc/carpeta_dosis/i for i in os.listdir(direc/carpeta_dosis)] # carpeta de distribusión de dosis
    estruc = sorted(estruc)
    dosis = sorted(dosis) # ordenar para que las posiciones correspondan entre las dos carpetas
    print("Carpetas cargadas")
    if len(estruc) != len(dosis):
        print("La cantidad de volúmenes no es equivalente a la cantidad de máscaras")
    print('La carpeta {} tiene {} volúmenes y la carpeta {} tiene {} máscaras'.format(carpeta_estruc, len(estruc),carpeta_dosis, len(dosis)))
    
    return estruc, dosis


# ## Datos de entrenamiento

# In[3]:


def traindata():
    estruc, dosis = cargar_carpetas('DatasetPI/EntrenamientoPI', 'Estructuras', 'Dosis')
    dataset = {"train" : Dataset(estruc[:180], dosis[:180]),"test" : Dataset(estruc[180:], dosis[180:])}
    print(f'La cantidad de imágenes de entrenamiento son {len(dataset["train"])} y la cantidad de validación son {len(dataset["test"])}.')
    dataloader = {"train" : torch.utils.data.DataLoader(dataset["train"], batch_size=2, shuffle=True, pin_memory=True),
                  "test" : torch.utils.data.DataLoader(dataset["test"], batch_size=2, pin_memory=True)}

    imges, maskes = next(iter(dataloader["train"]))
    imges.shape, maskes.shape
    return dataloader, dataset


# ## Datos de testeo

# In[4]:


def testdata():
    estruc, dosis = cargar_carpetas('DatasetPI/TesteoPI', 'Estructuras', 'Dosis')
    n = str(dosis[3]).split("/")[-1].split(".")[0]
    lista=[]
    for i in range(len(dosis)):
        lista.append(str(dosis[i]).split("/")[-1].split(".")[0])
    dataset = {"test" : Dataset(estruc[:], dosis[:])}
    print(f'La cantidad de imágenes de testeo son {len(dataset["test"])}.')
    dataloader = {"test" : torch.utils.data.DataLoader(dataset["test"], batch_size=2, pin_memory=True)}
    imges, maskes = next(iter(dataloader["test"]))
    imges.shape, maskes.shape
    return dataloader, dataset, lista


# # Dataloader

# In[5]:


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# In[6]:


class Dataset (torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y # dupla
       
    def __len__(self):
        assert len(self.x) == len(self.y), "La cantidad de volúmenes no es equivalente a la cantidad de máscaras"
        return len(self.x) # cantidad de ejemplos
    
    
    def __getitem__(self,ix):
        img = np.load(self.x[ix]) # carga de ejemplos
        mask = np.load(self.y[ix])
        img_oh = (np.arange(4) == img[...,None]).astype(np.float64) # one hot encoding de las estructuras
        img_tensor= torch.tensor(img_oh).permute(3, 0, 1, 2) # poner como primera dimensión la cantidad el tipo de órganos 
        mask_e = mask/36.25 # dosis escaladas
        mask_tensor = torch.tensor(mask_e).unsqueeze(0) # agregar la dimensión equivalente al one hot
        
        return img_tensor, mask_tensor


# # Métricas

# ## Dice

# In[7]:


def dice(true, pred, smooth=0.0000001):
    true=torch.round(true, decimals=1) # se redondea con un decimal los valores de los pixeles para que las coincidencias no sea tan excasas 
    pred=torch.round(pred, decimals=1) # como lo son cuando se tienen 6 valores despues de la coma.
    intersect = torch.sum((true == pred).int()) # la intersección son los pixeles en donde coincide el valor en los dos ejemplos 
    total = math.prod(true.shape) + math.prod(pred.shape) # el total es la suma de todos los pixeles que forman ambos ejemplos
    return   torch.mean((2 * intersect) /(total +smooth)) 


# ## IoU

# In[8]:


def iou(true, pred, smooth=0.0000001):
    true=torch.round(true, decimals=1)
    pred=torch.round(pred, decimals=1)
    intersect = torch.sum((true == pred).int())
    union = math.prod(true.shape) + math.prod(pred.shape)-intersect
    return   torch.mean(intersect /(union+smooth))


# # Utils

# ## Calculo de tiempo

# In[9]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins*60))
    
    return elapsed_mins, elapsed_secs


# ## Seed

# In[10]:


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# # U-Net Básica

# In[11]:


def conv3x3_bn(ci, co):
    return torch.nn.Sequential(
        torch.nn.Conv3d(ci, co, 3, padding=1),
        torch.nn.BatchNorm3d(co),
        torch.nn.ReLU(inplace=True)
    ) # bloque de convolución con kernel 3x3x3, batch normalization y activación ReLU

def encoder_conv(ci, co):
    return torch.nn.Sequential(
        torch.nn.MaxPool3d((2, 2, 2)),
        conv3x3_bn(ci, co),
        conv3x3_bn(co, co),
    ) # bloque maxpool con kernel 2x2x2 + convolución + convolución --> construye la bajada o el encoder

class deconv(torch.nn.Module):
    def __init__(self, ci, co):
        super(deconv, self).__init__()
        self.upsample = torch.nn.ConvTranspose3d(ci, co, 2, stride=2, padding=0) # convolución transpuesta de upsample
        self.conv1 = conv3x3_bn(ci, co)
        self.conv2 = conv3x3_bn(co, co)
    
    def forward(self, x1, x2): # recibe la salida de la capa anterior y la salida de la etapa correspondiente del encoder
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], axis=1) # concatena los dos mapas de características recibidos
        x = self.conv1(x) 
        x = self.conv2(x) # pasa por dos convoluciones la concatenación 
        return x

class UNetB(torch.nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()

        # lista de cantidad de filtros para convoluciones en encoder-decoder
        c = [64, 128, 256, 512]

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

        # útlima capa de convolución con kernel 1x1x1 que da la distribución de dosis 
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


# # U-Net Mejorada 

# In[12]:


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
class UNetM(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features=[64, 128, 256, 512], dropout=False,  upsampling=True):
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


# # ResU-Net

# In[13]:


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
        x = self.encoder.stem(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x = self.encoder.layer4(x3)
        x = self.deconv1(x, x3)
        x = self.deconv2(x, x2)
        x = self.deconv3(x, x1)
        x = self.out(x, x_in)
        return x


# # * Funcionamiento

# In[14]:


model = UNetB()
output = model(torch.randn((1, 4, 64, 64, 64)))
output.shape


# In[15]:


model = UNetM()
output = model(torch.randn((1, 4, 64, 64, 64)))
output.shape


# In[16]:


model = ResUNet()
output = model(torch.randn((1, 4, 64, 64, 64)))
output.shape


# # Entrenamiento

# In[14]:


def fitB(model, dataloader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    best_valid_loss = float("inf")
    create_dir("file")
    checkpoint_path = "file/checkpoint.pth"
    model.to(device)
    hist = {'loss': [], 'test_loss': []}
    met = {'iou': [], 'Dice': [], 'iou_test': [], 'Dice_test': []}
    start_time = time.time()
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        bar = tqdm(dataloader['train'])
        train_loss= []
        train_dice= []
        train_iou= []
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
            train_dice.append(dices.item())
            train_iou.append(ious.item())
            bar.set_description(f"loss: {np.mean(train_loss):.5f}, iou: {np.mean(train_iou):.5f}, dice: {np.mean(train_dice):.5f}")
        hist['loss'].append(np.mean(train_loss))
        met['Dice'].append(np.mean(train_dice))
        met['iou'].append(np.mean(train_iou))
        bar = tqdm(dataloader['test'])
        
        test_loss = []
        test_dice = []
        test_iou = []
        model.eval()
        with torch.no_grad():
            for imges, maskes in bar:
                imges, maskes = imges.to(device,dtype=torch.float), maskes.to(device, dtype=torch.float)
                y_hat = model(imges)
                loss = criterion(y_hat, maskes)
                ious = iou(y_hat, maskes)
                dices = dice(y_hat, maskes)
                test_loss.append(loss.item())
                test_dice.append(dices.item())
                test_iou.append(ious.item())
                bar.set_description(f"val_loss: {np.mean(test_loss):.5f}, iou: {np.mean(test_iou):.5f}, dice: {np.mean(test_dice):.5f}")
        hist['test_loss'].append(np.mean(test_loss))
        met['Dice_test'].append(np.mean(test_dice))
        met['iou_test'].append(np.mean(test_iou))
    
        if np.mean(test_loss) < best_valid_loss:
            data_str = f"El loss de validación mejoró de {best_valid_loss:.5f} a {loss:.5f}. Guardando checkpoint: {checkpoint_path}"
            print(data_str)
            best_valid_loss = np.mean(test_loss)
            torch.save(model.state_dict(), checkpoint_path)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    data_str = f' Entrenamiento de {epochs} épocas finalizado en {epoch_mins}m {epoch_secs}s\n'
    print()
    print(data_str)
    
    return hist, met


# In[15]:


def train(model, dataloader, hist, met, optimizer, loss_fn, device):
    epoch_loss = 0.0
    bar = tqdm(dataloader['train'])
    train_loss= []
    train_dice= []
    train_iou= []
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
        train_dice.append(dices.item())
        train_iou.append(ious.item())
        bar.set_description(f"loss: {np.mean(train_loss):.5f}, iou: {np.mean(train_iou):.5f}, dice: {np.mean(train_dice):.5f}")
    hist['loss'].append(np.mean(train_loss))
    met['Dice'].append(np.mean(train_dice))
    met['IoU'].append(np.mean(train_iou))
    
    return np.mean(train_loss), hist, met


# In[16]:


def evaluate(model, dataloader, hist, met, loss_fn, device):
    epoch_loss = 0.0
    bar = tqdm(dataloader['test'])
    test_loss = []
    test_dice = []
    test_iou = []
    model.eval()
    with torch.no_grad():
        for imges, maskes in bar:
            imges, maskes = imges.to(device,dtype=torch.float), maskes.to(device, dtype=torch.float)
            y_pred = model(imges)
            loss = loss_fn(y_pred, maskes)
            ious = iou(y_pred, maskes)
            dices = dice(y_pred, maskes)
            epoch_loss += loss.item()
            test_loss.append(loss.item())
            test_dice.append(dices.item())
            test_iou.append(ious.item())
            bar.set_description(f"val_loss: {np.mean(test_loss):.5f}, iou: {np.mean(test_iou):.5f}, dice: {np.mean(test_dice):.5f}")
        hist['test_loss'].append(np.mean(test_loss))
        met['Dice_test'].append(np.mean(test_dice))
        met['IoU_test'].append(np.mean(test_iou))
        
    return np.mean(test_loss), hist, met


# In[17]:


def fit(model, device, epochs=10, loss_fn = torch.nn.MSELoss()):
    if __name__ == "__main__":

        seeding(42)
        create_dir("file")

        epochs = epochs
        lr = 1e-4
        checkpoint_path = "file/checkpoint.pth" # Para guardar el modelo por si se deteniene el antrenamiento

        dataloaders = dataloader

        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, min_lr= 1e-8)
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
                data_str = f"El loss de validación mejoró de {best_valid_loss:.4f} a {valid_loss:.4f}. Guardando checkpoint: {checkpoint_path}"
                print(data_str)

                best_valid_loss = valid_loss
                torch.save(model.state_dict(), checkpoint_path)
        # Mostrar avance del entrenamiento    
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        data_str = f' Entrenamiento de {epochs} épocas finalizado en {epoch_mins}m {epoch_secs}s\n'
        print()
        print(data_str)
        
        return hist, met


# In[18]:


def traininfo(hist, met, red="U-Net Básica", epoca=500, loss="MSELoss"):
    
    epocas = np.round(np.arange(0, epoca+1, 1), decimals=2)
    
    dfh = pd.DataFrame(hist)
    dfm = pd.DataFrame(met)
    
    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(
        go.Scatter(x=epocas, y=dfh["loss"], name=("Pérdida en train"), marker=dict(color="orchid")),row=1, col=1)

    fig.add_trace(
        go.Scatter(x=epocas, y=dfh["test_loss"], name=("Pérdida en val"), marker=dict(color="darkorchid")),row=1, col=1)

    fig.add_trace(
        go.Scatter(x=epocas, y=dfm["Dice"], name=("Dice en train"), marker=dict(color="violet")),row=2, col=1)

    fig.add_trace(
        go.Scatter(x=epocas, y=dfm["Dice_test"], name=("Dice en val"), marker=dict(color="darkviolet")),row=2, col=1)

    fig.update_layout(title_text=f"Métricas del entrenamiento de {red} con {epoca} epocas y {loss}")
    fig.show()


# In[19]:


device = torch.device('cuda')


# # Guardado del modelo entrenado
# Guarda en un string de bites (archivo binario) un state dictionary que contiene las capas con los valores de paramentros aprendidos.

# In[20]:


def guardar(tipo, model, hist, met):
    
    create_dir("Modelos_entrenados")
    
    if tipo==1: checkpoint_path ="Modelos_entrenados/checkpoint_UB.pth"
    elif tipo==2: checkpoint_path ="Modelos_entrenados/checkpoint_UM.pth"
    elif tipo==3: checkpoint_path ="Modelos_entrenados/checkpoint_RU.pth"
    else: print("No corresponde a un tipo de red")
    
    torch.save({"model_state_dict": model.state_dict(),
                "hist": hist,
                "met": met},
               checkpoint_path)
    print("Modelo entrenado guardado")
    
def cargar(tipo, model):

    if tipo==1: 
        checkpoint_path ="Modelos_entrenados/checkpoint_UB.pth"
    elif tipo==2: 
        checkpoint_path ="Modelos_entrenados/checkpoint_UM.pth"
    elif tipo==3: 
        checkpoint_path ="Modelos_entrenados/checkpoint_RU.pth"
    else: print("No corresponde a un tipo de red")
        
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    hist=checkpoint['hist']
    met=checkpoint['met'] 
    
    print("Modelo cargado")
    
    return model, hist, met


# # Predicciones con dataset de testeo

# In[21]:


def predecir(direc):
    create_dir(direc)
    dataloader, dataset, lista = testdata()
    red= direc.split("/")[-1].split(".")[0]
    for i in range(len(dataset['test'])):
        model.eval()
        with torch.no_grad():
            img, mask = dataset['test'][i]
            nombre = "pred_"+red+"_"+lista[i]
            pred_mask= model(img.unsqueeze(0).to(device, dtype=torch.float))[0]
            print(f"{nombre} guardada en {direc}")
            pred_mask=pred_mask.cpu()
            np.save(f'{direc}/{nombre}',pred_mask)


# # Entrenamientos, guardado y predicciones

# ## U-Net Básica

# In[24]:


model = UNetB()
dataloader, dataset = traindata()
hist, met = fitB(model, dataloader, epochs=500)


# In[26]:


guardar(1, model, hist, met)


# In[22]:


model, hist, met=cargar(1, UNetB())


# In[23]:


traininfo(hist, met, red="U-Net Básica", epoca=500, loss="MSELoss")


# In[29]:


model.to(device)
for i in range(5):
    model.eval()
    with torch.no_grad():
        ix = random.randint(0, len(dataset['test'])-1)
        img, mask = dataset['test'][ix]
        pred_mask = model(img.unsqueeze(0).to(device, dtype=torch.float))[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.set_title('Objetivo', fontsize=30)
    ax1.imshow(mask[:,:,:,29].permute(1,2,0),cmap="magma")
    ax2.set_title('Predicción', fontsize=30)
    ax2.imshow(pred_mask.squeeze().cpu().numpy()[:,:,29],cmap="magma")
    plt.show()


# In[30]:


predecir("DatasetPI/TesteoPI/UB")


# ## U-Net Mejorada

# In[24]:


model = UNetM()
model = model.to(device)
dataloader, dataset = traindata()


# In[25]:


hist, met = fit(model= model, device= device, epochs=500, loss_fn = torch.nn.MSELoss())


# In[27]:


guardar(2, model, hist, met)


# In[24]:


model, hist, met=cargar(2, UNetM())


# In[25]:


traininfo(hist, met, red="U-Net Mejorada", epoca=500, loss="MSELoss")


# In[31]:


model.to(device)
for i in range(5):
    model.eval()
    with torch.no_grad():
        ix = random.randint(0, len(dataset['test'])-1)
        img, mask = dataset['test'][ix]
        pred_mask = model(img.unsqueeze(0).to(device, dtype=torch.float))[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.set_title('Objetivo', fontsize=30)
    ax1.imshow(mask[:,:,:,29].permute(1,2,0),cmap="magma")
    ax2.set_title('Predicción', fontsize=30)
    ax2.imshow(pred_mask.squeeze().cpu().numpy()[:,:,29],cmap="magma")
    plt.show()


# In[30]:


predecir("DatasetPI/TesteoPI/UM")


# ## ResU-Net

# In[22]:


model = ResUNet()
model = model.to(device)
dataloader, dataset = traindata()


# In[23]:


hist, met = fit(model= model, device = device, epochs=500, loss_fn = torch.nn.MSELoss())


# In[24]:


guardar(3, model, hist, met)


# In[26]:


model, hist, met=cargar(3, ResUNet())


# In[27]:


traininfo(hist, met, red="ResU-Net", epoca=500, loss="MSELoss")


# In[29]:


model.to(device)
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


# In[26]:


predecir("DatasetPI/TesteoPI/RU")


# # Análisis de resultados
# Para comparar los resultados conseguidos con las tres redes se realizan dos tipos de gráficas

# ## Carga de ejemplos

# In[28]:


def cargar_carpeta(carpeta, tipo="paciente", graph=False):
    direc = Path(carpeta +"/")
    ejemplos = [direc/i for i in os.listdir(direc)]
    ejemplos = sorted(ejemplos)
    if graph==True:
        print(f"Ejemplos de {tipo} cargados")
    return ejemplos

def cargar_ejem (dir_est, dir_targets, dirpredUB, dirpredUM, dirpredRU, i, graph=True):
    
    
    ests = cargar_carpeta(dir_est, "estructuras")
    est = np.load(ests[i])
    
    targets = cargar_carpeta(dir_targets, "dosis objetivos")
    target = np.load(targets[i])
    target = target/36.25
    
    results1 = cargar_carpeta(dirpredUB, "predicciones de U-Net Básica",graph)
    pred1 = np.load(results1[i]).squeeze().astype('float64')
    
    results2 = cargar_carpeta(dirpredUM, "predicciones de U-Net Mejorada",graph)
    pred2 = np.load(results2[i]).squeeze().astype('float64')
    
    results3 = cargar_carpeta(dirpredRU, "predicciones de ResU-Net",graph)
    pred3 = np.load(results3[i]).squeeze().astype('float64')
    
    if graph==True:
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,10))
        sns.set_style("dark")
        ax1.imshow(target[:,:,32],cmap="magma")
        ax2.imshow(pred1[:,:,32],cmap="magma")
        ax3.imshow(pred2[:,:,32],cmap="magma")
        ax4.imshow(pred3[:,:,32],cmap="magma")
        ax1.set_title("Objetivo de distribución de dosis")
        ax2.set_title("Predicción de U-Net Básica")
        ax3.set_title("Predicción de U-Net Mejorada")
        ax4.set_title("Predicción de ResU-Net")
        plt.show()
    
    return est, target, pred1, pred2, pred3


# In[30]:


est, target, pred1, pred2, pred3 = cargar_ejem('DatasetPI/TesteoPI/Estructuras',
                                               'DatasetPI/TesteoPI/Dosis',
                                               'DatasetPI/TesteoPI/UB',
                                               'DatasetPI/TesteoPI/UM',
                                               'DatasetPI/TesteoPI/RU',
                                                i=16)


# ## Coeficiente de Dice para cada intervalo porcentual de dosis 
# En esta grafica se compara los tres modelos entre si diferenciando su capacidad de predecir cada valor de porcentaje de dosis por separado. Primero se elige un ejemplo y se realiza su predicción en cada modelo, luego genera un data frame que guarda los datos del coeficiente de Dice entre cada predicción y el objetivo para cada cantidad de dosis porcentual. Finalmente se muestra la gráfica en la que el eje y es el valor Dice alcanzado y el x es el porcentaje de dosis particular.

# In[31]:


def valores(tru, pre):
    perc=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
    suma=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
    for i in range(len(perc)):
        interseccion =np.count_nonzero(((pre == perc[i])*(pre == tru))) #cantidad de pixeles que coinciden entre la predicción y el objetivo y que son iguales a perc[i](un porcentaje determinado) 
        union=np.count_nonzero(tru == perc[i])+np.count_nonzero(pre == perc[i])#count_nonzer cuenta los vóxeles en los que el valor booleano coincide y es True (descarta las coincidencias False)
        suma[i] = (2*interseccion)/(union+0.000000000000000000000001) #
    return suma

def dataframe(suma0, suma1, suma2, suma3):
    tipos=["Objetivo", "U-Net Básica", "U-Net Mejorada", "ResU-Net"]
    t=["O", "UB", "UM", "RU"]
    red=[]
    r=[]
    for x in range(len(tipos)):
        for i in range(13):
            red.append(tipos[x])
            r.append(t[x]+"_"+str(np.round(i*0.1, decimals=2)))
            
    perc=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    perc=perc*len(tipos)
    sumar=suma0+suma1+suma2+suma3
    data = {'Red': red,
            'Dice score':sumar,
            "Porcentajes de dosis [%]":perc
            }
    df = pd.DataFrame(data, index= r)
    return df

def Dice_porcent(true, pred1, pred2, pred3, graph=True): 
    
    print
    true = torch.tensor(true)
    true = torch.round(true, decimals=1)

    pre1 = torch.tensor(pred1)
    pred1 = torch.round(pre1, decimals=1)

    pre2 = torch.tensor(pred2)
    pred2 = torch.round(pre2, decimals=1)
    
    pre3 = torch.tensor(pred3)
    pred3 = torch.round(pre3, decimals=1)
    
    suma0 = valores(true, true)
    suma1 = valores(true, pred1)
    suma2 = valores(true, pred2)
    suma3 = valores(true, pred3)
    
    df = dataframe(suma0, suma1, suma2, suma3)
    
    if graph==True:
    
        tipos=["Objetivo", "U-Net Básica", "U-Net Mejorada", "ResU-Net"]
        fig = px.line(df, x='Porcentajes de dosis [%]', y='Dice score', color='Red', color_discrete_map={tipos[0]: "darkblue",
                         tipos[1]: "darkviolet", tipos[2]:"violet",  tipos[3]:"mediumpurple"}, markers=True)
        fig.update_layout(title='Dice score de cada porcentaje de dosis')
        fig.update_traces(textposition="bottom right")
        fig.show()
    
    return df


# In[32]:


df = Dice_porcent(target, pred1, pred2, pred3) 
display(df)


# ## Histograma Dosis-Volumen 
# 
# En el DVH se grafica la distribución de frecuencia de dosis acumulada de cada órgano. En RT son una herramienta útil para comparar diferentes planes de tratamiento en un paciente. Básicamente en el eje y se tiene el porcentaje de cantidad de voxeles total que corresponden a un organo y en el eje x se tiene el porcentaje de dosis considerando el 100% la dosis prescripta para el PTV. Entonces para cada valor de dosis hay un valor de porcentaje de cada órgano que recibe dicha dosis o más.
# 

# ### Primero se identifican los órganos

# In[33]:


def onehot(est):
    esto = (np.arange(4) == est[...,None]).astype(np.float64)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,20))
    ax1.imshow(esto[:,:,32,0],cmap="magma")
    ax2.imshow(esto[:,:,32,1],cmap="magma")
    ax3.imshow(esto[:,:,32,2],cmap="magma")
    ax4.imshow(esto[:,:,32,3],cmap="magma")
    plt.show()
    return esto


# ### Segundo se realizan histogramas de cada órgano en cada predicción

# In[39]:


def HistDV (est, dos, pred1, pred2, pred3, rec, ptv, vej, ejes=True):
    
    dos = torch.from_numpy(dos)
    pred1 = torch.from_numpy(pred1)
    pred2 = torch.from_numpy(pred2)
    pred3 = torch.from_numpy(pred3)
    
    est = torch.from_numpy(est)
    estrec = est[:,:,:,rec]
    estptv = est[:,:,:,ptv]
    estvej = est[:,:,:,vej]
    rec=(dos*estrec).view(-1)
    ptv=(dos*estptv).view(-1)
    vej=(dos*estvej).view(-1)
    rec= rec[rec!=0]
    ptv= ptv[ptv!=0]
    vej= vej[vej!=0]
    
    recp1=(pred1*estrec).view(-1)
    ptvp1=(pred1*estptv).view(-1)
    vejp1=(pred1*estvej).view(-1)
    recp1= recp1[recp1!=0]
    ptvp1= ptvp1[ptvp1!=0]
    vejp1= vejp1[vejp1!=0]
    
    recp2=(pred2*estrec).view(-1)
    ptvp2=(pred2*estptv).view(-1)
    vejp2=(pred2*estvej).view(-1)
    recp2= recp2[recp2!=0]
    ptvp2= ptvp2[ptvp2!=0]
    vejp2= vejp2[vejp2!=0]
    
    recp3=(pred3*estrec).view(-1)
    ptvp3=(pred3*estptv).view(-1)
    vejp3=(pred3*estvej).view(-1)
    recp3= recp3[recp3!=0]
    ptvp3= ptvp3[ptvp3!=0]
    vejp3= vejp3[vejp3!=0]
    
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(20,12))
    sns.set(style="darkgrid")
    sns.histplot( rec, ax=ax1, kde=True, bins=100, color="darkblue", element='step', stat = 'probability')
    sns.histplot( ptv, ax=ax5, kde=True, bins=100, color="darkblue", element='step', stat = 'probability')
    sns.histplot( vej, ax=ax9, kde=True, bins=100, color="darkblue", element='step', stat = 'probability')
    ax1.set_title("Objetivo: Dosis en el recto", fontsize=12)
    ax5.set_title("Objetivo: Dosis en el PTV", fontsize=12)
    ax9.set_title("Objetivo: Dosis en la vejiga", fontsize=12)
    sns.histplot( recp1, ax=ax2, kde=True, bins=100, color="darkviolet", element='step', stat = 'probability')
    sns.histplot( ptvp1, ax=ax6, kde=True, bins=100, color="darkviolet", element='step', stat = 'probability')
    sns.histplot( vejp1, ax=ax10, kde=True, bins=100, color="darkviolet", element='step', stat = 'probability')
    ax2.set_title("U-Net Básica: Dosis en el recto", fontsize=12)
    ax6.set_title("U-Net Básica: Dosis en el PTV", fontsize=12)
    ax10.set_title("U-Net Básica: Dosis en la vejiga", fontsize=12)
    sns.histplot( recp2, ax=ax3, kde=True, bins=100, color="violet", element='step', stat = 'probability')
    sns.histplot( ptvp2, ax=ax7, kde=True, bins=100, color="violet", element='step', stat = 'probability')
    sns.histplot( vejp2, ax=ax11, kde=True, bins=100, color="violet", element='step', stat = 'probability')
    ax3.set_title("U-Net Mejorada: Dosis en el recto", fontsize=12)
    ax7.set_title("U-Net Mejorada: Dosis en el PTV", fontsize=12)
    ax11.set_title("U-Net Mejorada: Dosis en la vejiga", fontsize=12)
    sns.histplot( recp3, ax=ax4, kde=True, bins=100, color="mediumpurple", element='step', stat = 'probability')
    sns.histplot( ptvp3, ax=ax8, kde=True, bins=100, color="mediumpurple", element='step', stat = 'probability')
    sns.histplot( vejp3, ax=ax12, kde=True, bins=100, color="mediumpurple", element='step', stat = 'probability')
    ax4.set_title("ResU-Net: Dosis en el recto", fontsize=12)
    ax8.set_title("ResU-Net: Dosis en el PTV", fontsize=12)
    ax12.set_title("ResU-Net: Dosis en la vejiga", fontsize=12)
    
    if ejes==True:
        recm=0.06
        ptvm=0.1
        vegm=0.09
        ax1.set_ylim(0,recm)
        ax2.set_ylim(0,recm)
        ax3.set_ylim(0,recm)
        ax4.set_ylim(0,recm)

        ax5.set_ylim(0,ptvm)
        ax6.set_ylim(0,ptvm)
        ax7.set_ylim(0,ptvm)
        ax8.set_ylim(0,ptvm)

        ax9.set_ylim(0,vegm)
        ax10.set_ylim(0,vegm)
        ax11.set_ylim(0,vegm)
        ax12.set_ylim(0,vegm)
    
    return rec, ptv, vej, recp1, ptvp1, vejp1, recp2, ptvp2, vejp2, recp3, ptvp3, vejp3 


# In[35]:


est = onehot(est)


# In[40]:


# Los últimos tres argumentos para la función HistDV son la posición del recto, ptv y vejiga (en ese orden) en el onehot,
# en todos los ejemplos esas tres estructuras aparecen en ese orden de izquierda a derecha y en la altura del corte elegido
# (24, se puede ver el orden en la ´primera imagen que es el fondo con la sombra de todas las estructuras extraídas)
# no se suele ver la vejiga, así que la imagen en negro corresponde a esta, además de que el recto suele ser más pequeño
# que la próstata. 
recto = int (input('¿Qué capa es el recto?'))
ptvs = int (input('¿Qué capa es la próstata?'))
vejiga = int (input('¿Qué capa es la vejiga?'))
rec, ptv, vej, recp1, ptvp1, vejp1, recp2, ptvp2, vejp2, recp3, ptvp3, vejp3 = HistDV(est, target, pred1, pred2, pred3, recto, ptvs, vejiga)


# In[41]:


def HistDV2 (est, dos, pred1, pred2, pred3, rec, ptv, vej, ejes=True):
    
    dos = torch.from_numpy(dos)
    pred1 = torch.from_numpy(pred1)
    pred2 = torch.from_numpy(pred2)
    pred3 = torch.from_numpy(pred3)
    
    est = torch.from_numpy(est)
    estrec = est[:,:,:,rec]
    estptv = est[:,:,:,ptv]
    estvej = est[:,:,:,vej]
    rec=(dos*estrec).view(-1)
    ptv=(dos*estptv).view(-1)
    vej=(dos*estvej).view(-1)
    rec= rec[rec!=0]
    ptv= ptv[ptv!=0]
    vej= vej[vej!=0]
    
    recp1=(pred1*estrec).view(-1)
    ptvp1=(pred1*estptv).view(-1)
    vejp1=(pred1*estvej).view(-1)
    recp1= recp1[recp1!=0]
    ptvp1= ptvp1[ptvp1!=0]
    vejp1= vejp1[vejp1!=0]
    
    recp2=(pred2*estrec).view(-1)
    ptvp2=(pred2*estptv).view(-1)
    vejp2=(pred2*estvej).view(-1)
    recp2= recp2[recp2!=0]
    ptvp2= ptvp2[ptvp2!=0]
    vejp2= vejp2[vejp2!=0]
    
    recp3=(pred3*estrec).view(-1)
    ptvp3=(pred3*estptv).view(-1)
    vejp3=(pred3*estvej).view(-1)
    recp3= recp3[recp3!=0]
    ptvp3= ptvp3[ptvp3!=0]
    vejp3= vejp3[vejp3!=0]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20,13))
    sns.set(style="darkgrid")
    sns.histplot( rec, ax=ax1, kde=True, bins=100, color="darkblue", element='step', stat = 'probability')
    sns.histplot( ptv, ax=ax2, kde=True, bins=100, color="darkblue", element='step', stat = 'probability')
    sns.histplot( vej, ax=ax3, kde=True, bins=100, color="darkblue", element='step', stat = 'probability')
    sns.histplot( recp1, ax=ax1, kde=True, bins=100, color="darkviolet", element='step', stat = 'probability')
    sns.histplot( ptvp1, ax=ax2, kde=True, bins=100, color="darkviolet", element='step', stat = 'probability')
    sns.histplot( vejp1, ax=ax3, kde=True, bins=100, color="darkviolet", element='step', stat = 'probability')
    sns.histplot( recp2, ax=ax1, kde=True, bins=100, color="violet", element='step', stat = 'probability')
    sns.histplot( ptvp2, ax=ax2, kde=True, bins=100, color="violet", element='step', stat = 'probability')
    sns.histplot( vejp2, ax=ax3, kde=True, bins=100, color="violet", element='step', stat = 'probability')
    sns.histplot( recp3, ax=ax1, kde=True, bins=100, color="mediumpurple", element='step', stat = 'probability')
    sns.histplot( ptvp3, ax=ax2, kde=True, bins=100, color="mediumpurple", element='step', stat = 'probability')
    sns.histplot( vejp3, ax=ax3, kde=True, bins=100, color="mediumpurple", element='step', stat = 'probability')
    ax1.set_title("Dosis en el recto. Azul:objetivo, Morado:U-Net B, Rosado: U-Net M y Lila: ResU-Net", fontsize=20)
    ax2.set_title("Dosis en el PTV. Azul:objetivo, Morado:U-Net B, Rosado: U-Net M y Lila: ResU-Net", fontsize=20)
    ax3.set_title("Dosis en la vejiga. Azul:objetivo, Morado:U-Net B, Rosado: U-Net M y Lila: ResU-Net", fontsize=20)
    
    return rec, ptv, vej, recp1, ptvp1, vejp1, recp2, ptvp2, vejp2, recp3, ptvp3, vejp3 


# In[42]:


rec, ptv, vej, recp1, ptvp1, vejp1, recp2, ptvp2, vejp2, recp3, ptvp3, vejp3 = HistDV2(est, target, pred1, pred2, pred3, recto, ptvs, vejiga)


# ### Por ultimo se obtien el HDV

# In[43]:


def acumular(organo):
    
    porcentajes = np.round(np.arange(0, 1.21, 0.01), decimals=2)
    acumulado = np.zeros(len(porcentajes))
    
    org = torch.round(organo, decimals=2)
    serie = pd.Series(org) # convertir los datos del grafico en una tabla 
    df = pd.DataFrame(serie.value_counts(normalize = True, sort=False)) # cuenta la ocurrencia de cada dosis y las ordena 
    df = df.rename_axis('Dosis').reset_index().sort_values(by=["Dosis"]) # define la columna dosis
    ndf= df.to_numpy()

    voxeles = np.zeros(len(porcentajes))
    for i in range(len(porcentajes)): 
        for j in range(len(ndf)):
             if ndf[j,0]==porcentajes[i]:
                    voxeles[i]=ndf[j,1] # rellena de 0 los valores de dosis faltantes 

        for i in range(len(acumulado)):
            acumulado[i] = sum(voxeles[i:]) # realiza el acumulado del DVH     
        
    return acumulado



def DVH (ptv,ptvp1, ptvp2, ptvp3, rec, recp1, recp2, recp3, vej, vejp1, vejp2, vejp3):
 
    organos = [ptv,ptvp1, ptvp2, ptvp3, rec, recp1, recp2, recp3, vej, vejp1, vejp2, vejp3]
    organo = ["ptv_objetivo", "ptv_U-Net_B", "ptv_U-Net_M", "ptv_U-ResNet", "recto_objetivo", "recto_U-Net_B", "recto_U-Net_M", "recto_ResU-Net", "vejiga_objetivo", "vejiga_U-Net_B", "vejiga_U-Net_M", "vejiga_ResU-Net"]
    a=np.zeros(121)
    grafico=[a,a,a,a,a,a,a,a,a,a,a,a]
    df = pd.DataFrame ({'Porcentajes de dosis [%]': np.arange(0, 121, 1)})
    
    for x in range(len(organos)):        
        df [organo[x]] = acumular(organos[x]).tolist()
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["recto_objetivo"], name="recto_objetivo", line=dict(color='darkblue', dash='solid')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["ptv_objetivo"], name="ptv_objetivo", line=dict(color='darkblue', dash='dashdot')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["vejiga_objetivo"], name="vejiga_objetivo", line=dict(color='darkblue', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["recto_U-Net_B"], name="recto_U-Net_B", line=dict(color='darkviolet', dash='solid')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["ptv_U-Net_B"], name="ptv_U-Net_B", line=dict(color='darkviolet', dash='dashdot')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["vejiga_U-Net_B"], name="vejiga_U-Net_B", line=dict(color='darkviolet', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["recto_U-Net_M"], name="recto_U-Net_M", line=dict(color='violet', dash='solid')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["ptv_U-Net_M"], name="ptv_U-Net_M", line=dict(color='violet', dash='dashdot')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["vejiga_U-Net_M"], name="vejiga_U-Net_M", line=dict(color='violet', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["recto_ResU-Net"], name="recto_ResU-Net", line=dict(color='mediumpurple', dash='solid')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["ptv_U-ResNet"], name="ptv_U-ResNet", line=dict(color='mediumpurple', dash='dashdot')))
    fig.add_trace(go.Scatter(x=df['Porcentajes de dosis [%]'], y=df["vejiga_ResU-Net"], name="vejiga_ResU-Net", line=dict(color='mediumpurple', dash='dash')))
    fig.update_layout(title='DVH acumulado', xaxis_title='Porcentaje de dosis [%]', yaxis_title='Cantidad relativa de tejido')
    
    fig.show()


# In[44]:


DVH(ptv,ptvp1, ptvp2, ptvp3, rec, recp1, recp2, recp3, vej, vejp1, vejp2, vejp3)


# # índice de Dice final

# # Predicciones de los otros ejemplos 

# In[46]:


def ejemplos():    
    for i in range(30):

        sns.set_style("dark")

        est, target, pred1, pred2, pred3 = cargar_ejem('DatasetPI/TesteoPI/Estructuras',
                                                       'DatasetPI/TesteoPI/Dosis',
                                                       'DatasetPI/TesteoPI/UB',
                                                       'DatasetPI/TesteoPI/UM',
                                                       'DatasetPI/TesteoPI/RU',
                                                        i=i)
        print("----------------------------------------------------------------------")
        print(f"Ejemplo número {i}:")
        print()
        df = Dice_porcent(target, pred1, pred2, pred3)
        est = onehot(est)
        recto = int (input('¿Qué capa es el recto?'))
        ptv = int (input('¿Qué capa es la próstata?'))
        vejiga = int (input('¿Qué capa es la vejiga?'))
        si = input('¿Seguro?')
        if si!='si':
            recto = int (input('¿Qué capa es el recto?'))
            ptv = int (input('¿Qué capa es la próstata?'))
            vejiga = int (input('¿Qué capa es la vejiga?')) 

        rec, ptv, vej, recp1, ptvp1, vejp1, recp2, ptvp2, vejp2, recp3, ptvp3, vejp3 = HistDV2(est, target, pred1, pred2, pred3, recto, ptv, vejiga, ejes=False)
        DVH(ptv,ptvp1, ptvp2, ptvp3, rec, recp1, recp2, recp3, vej, vejp1, vejp2, vejp3)



# In[45]:


ejemplos()


# ## Coeficiente de Dice promedio para cada intervalo porcentual de dosis 

# In[47]:


def Dice_final():
    
    val0UB = []
    val10UB = []
    val20UB = []
    val30UB = []
    val40UB = []
    val50UB = []
    val60UB = []
    val70UB = []
    val80UB = []
    val90UB = []
    val100UB = []
    val110UB = []
    val0UM = []
    val10UM = []
    val20UM = []
    val30UM = []
    val40UM = []
    val50UM = []
    val60UM = []
    val70UM = []
    val80UM = []
    val90UM = []
    val100UM = []
    val110UM = []        
    val0RU = []
    val10RU = []
    val20RU = []
    val30RU = []
    val40RU = []
    val50RU = []
    val60RU = []
    val70RU = []
    val80RU = []
    val90RU = []
    val100RU = []
    val110RU = []
    
    for i in range(30):
        est, true, pred1, pred2, pred3 = cargar_ejem('DatasetPI/TesteoPI/Estructuras',
                                               'DatasetPI/TesteoPI/Dosis',
                                               'DatasetPI/TesteoPI/UB',
                                               'DatasetPI/TesteoPI/UM',
                                               'DatasetPI/TesteoPI/RU',
                                                i=i,
                                                graph=False)
        
        df = Dice_porcent(true, pred1, pred2, pred3, graph=False)
        val0UB.append(df['Dice score']['UB_0.0']) # en cada lista se van a tener los 30 calculos de índice de Dice para cada dosis de cada red
        val10UB.append(df['Dice score']['UB_0.1']) # se intento genrar con un loop pero los valores se reescribian en el dataframe
        val20UB.append(df['Dice score']['UB_0.2'])
        val30UB.append(df['Dice score']['UB_0.3'])
        val40UB.append(df['Dice score']['UB_0.4'])
        val50UB.append(df['Dice score']['UB_0.5'])
        val60UB.append(df['Dice score']['UB_0.6'])
        val70UB.append(df['Dice score']['UB_0.7'])
        val80UB.append(df['Dice score']['UB_0.8'])
        val90UB.append(df['Dice score']['UB_0.9'])
        val100UB.append(df['Dice score']['UB_1.0'])
        val110UB.append(df['Dice score']['UB_1.1'])
        
        val0UM.append(df['Dice score']['UM_0.0'])
        val10UM.append(df['Dice score']['UM_0.1'])
        val20UM.append(df['Dice score']['UM_0.2'])
        val30UM.append(df['Dice score']['UM_0.3'])
        val40UM.append(df['Dice score']['UM_0.4'])
        val50UM.append(df['Dice score']['UM_0.5'])
        val60UM.append(df['Dice score']['UM_0.6'])
        val70UM.append(df['Dice score']['UM_0.7'])
        val80UM.append(df['Dice score']['UM_0.8'])
        val90UM.append(df['Dice score']['UM_0.9'])
        val100UM.append(df['Dice score']['UM_1.0'])
        val110UM.append(df['Dice score']['UM_1.1'])
        
        val0RU.append(df['Dice score']['RU_0.0'])
        val10RU.append(df['Dice score']['RU_0.1'])
        val20RU.append(df['Dice score']['RU_0.2'])
        val30RU.append(df['Dice score']['RU_0.3'])
        val40RU.append(df['Dice score']['RU_0.4'])
        val50RU.append(df['Dice score']['RU_0.5'])
        val60RU.append(df['Dice score']['RU_0.6'])
        val70RU.append(df['Dice score']['RU_0.7'])
        val80RU.append(df['Dice score']['RU_0.8'])
        val90RU.append(df['Dice score']['RU_0.9'])
        val100RU.append(df['Dice score']['RU_1.0'])
        val110RU.append(df['Dice score']['RU_1.1'])

    val0UB = statistics.mean(val0UB)
    val10UB = statistics.mean(val10UB)
    val20UB = statistics.mean(val20UB)
    val30UB = statistics.mean(val30UB)
    val40UB = statistics.mean(val40UB)
    val50UB = statistics.mean(val50UB)
    val60UB = statistics.mean(val60UB)
    val70UB = statistics.mean(val70UB)
    val80UB = statistics.mean(val80UB)
    val90UB = statistics.mean(val90UB)
    val100UB = statistics.mean(val100UB)
    val110UB = statistics.mean(val110UB)
    val0UM = statistics.mean(val0UM)
    val10UM = statistics.mean(val10UM)
    val20UM = statistics.mean(val20UM)
    val30UM = statistics.mean(val30UM)
    val40UM = statistics.mean(val40UM)
    val50UM = statistics.mean(val50UM)
    val60UM = statistics.mean(val60UM)
    val70UM = statistics.mean(val70UM)
    val80UM = statistics.mean(val80UM)
    val90UM = statistics.mean(val90UM)
    val100UM = statistics.mean(val100UM)
    val110UM = statistics.mean(val110UM)        
    val0RU = statistics.mean(val0RU)
    val10RU = statistics.mean(val10RU)
    val20RU = statistics.mean(val20RU)
    val30RU = statistics.mean(val30RU)
    val40RU = statistics.mean(val40RU)
    val50RU = statistics.mean(val50RU)
    val60RU = statistics.mean(val60RU)
    val70RU = statistics.mean(val70RU)
    val80RU = statistics.mean(val80RU)
    val90RU = statistics.mean(val90RU)
    val100RU = statistics.mean(val100RU)
    val110RU = statistics.mean(val110RU)

    tipos=[ "U-Net Básica", "U-Net Mejorada", "ResU-Net"]
    t=["UB", "UM", "RU"]
    red=[]
    r=[]
    for x in range(len(tipos)):
        for i in range(12):
            red.append(tipos[x])
            r.append(t[x]+"_"+str(np.round(i*0.1, decimals=2)))
            
    perc=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    perc=perc*len(tipos)
    sumar = [val0UB, val10UB, val20UB, val30UB, val40UB, val50UB, val60UB, val70UB, val80UB, val90UB, val100UB, val110UB, 
             val0UM, val10UM, val20UM, val30UM, val40UM, val50UM, val60UM, val70UM, val80UM, val90UM, val100UM, val110UM, 
             val0RU, val10RU, val20RU, val30RU, val40RU, val50RU, val60RU, val70RU, val80RU, val90RU, val100RU, val110RU]
    data = {'Red': red,
            'Dice score':sumar,
            "Porcentajes de dosis [%]":perc
            }
    dff = pd.DataFrame(data, index= r)
    
    tipos=["U-Net Básica", "U-Net Mejorada", "ResU-Net"]
    fig = px.line(dff, x='Porcentajes de dosis [%]', y='Dice score', color='Red', color_discrete_map={ tipos[0]: "darkviolet", 
                tipos[1]:"violet",  tipos[2]:"mediumpurple"}, markers=True)
    fig.update_layout(title='Dice score de cada porcentaje de dosis')
    fig.update_traces(textposition="bottom right")
    fig.show()
    
    print(f"En promedio entre los 30 ejemplos, la red U-Net Básica tiene un índice Dice de {val0UB:.4f} para dosis del 0%")
    print(f"En promedio entre los 30 ejemplos, la red U-Net Básica tiene un índice Dice de {val50UB:.4f} para dosis del 50%")
    print(f"En promedio entre los 30 ejemplos, la red U-Net Básica tiene un índice Dice de {val100UB:.4f} para dosis del 100%")
    print(f"En promedio entre los 30 ejemplos, la red U-Net Mejorada tiene un índice Dice de {val0UM:.4f} para dosis del 0%")
    print(f"En promedio entre los 30 ejemplos, la red U-Net Mejorada tiene un índice Dice de {val50UM:.4f} para dosis del 50%")
    print(f"En promedio entre los 30 ejemplos, la red U-Net Mejorada tiene un índice Dice de {val100UM:.4f} para dosis del 100%")
    print(f"En promedio entre los 30 ejemplos, la red ResU-Net tiene un índice Dice de {val0RU:.4f} para dosis del 0%")
    print(f"En promedio entre los 30 ejemplos, la red ResU-Net tiene un índice Dice de {val50RU:.4f} para dosis del 50%")
    print(f"En promedio entre los 30 ejemplos, la red ResU-Net tiene un índice Dice de {val100RU:.4f} para dosis del 100%")

    return 


# In[48]:


Dice_final()  

