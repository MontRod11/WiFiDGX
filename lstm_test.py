#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:27:47 2021

@author: laura
"""

import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt
import random
import re
import haversine as hs
from haversine import Unit

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from torch.utils.data.sampler import Sampler
from prepare_data import separate_data, min_max_norm, min_max_norm_test, coordenates_norm
from prepare_data import coordenates_norm_test, coordenates_denorm
from prepare_OverlapSequences import completeseq_consolape, num_seq_tot

# In[2]: Semilla para reproducibilidad
seed = 15
torch.manual_seed(seed)   
torch.use_deterministic_algorithms(True)
random.seed(seed)
# torch.set_deterministic(True) 
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# CUDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# In[3]: Uso de GPU o CPU (is cuda available?)
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()
print(device)

# In[3.1]:Valores áximos y mínimos para la normalización de los datos de test:
    
# Apertura del documento que contiene los datos del conjunto de entrenamiento
coordenadas_train = pd.read_csv("/home/laura/TrayectoriasLSTM/datasets/datasets_degradacion_v1/train/conjuntoTrain_numTrayectorias9_Nexus_difGPSfreqcon0_numTrayectoriasTest225_numTrayectoriasValid2.csv",header=None) 
txt = 'conjuntoTrain_numTrayectorias9_Nexus_difGPSfreqcon0_numTrayectoriasTest225_numTrayectoriasValid2.csv'
numeros_nombre_train = [float(s) for s in re.findall(r'-?\d+\.?\d*', txt)]
numTrayect = int(numeros_nombre_train[0])

# Separación de los datos de entrada de los 'target', datos objetivo. 
trayect_data_train, trayect_labels_train = separate_data(coordenadas_train)

# Obtención de los valores máximos y mínimos de los datos de entrenamiento
trayect_data_train_normalized, minvalue, maxvalue = min_max_norm(trayect_data_train)
# Obtención de los valores máximos y mínimos de las coordenadas (target) de entrenamiento
trayect_labels_train_norm = np.zeros(trayect_labels_train.shape)
trayect_labels_train_norm[:,0], trayect_labels_train_norm[:,1], minmaxlat_train, minmaxlon_train = coordenates_norm(trayect_labels_train)

txt_valid = 'conjuntoValid_numTrayectorias9_numTrayectoriasValid2_Nexus_difGPSfreqcon0_numTrayectoriasTest225.csv'
numeros_nombre_valid = [float(s) for s in re.findall(r'-?\d+\.?\d*', txt_valid)]
numTrayects_valid = int(numeros_nombre_valid[1])

# In[3.2]: Preprocesado de los datos de test, separación entre datos-etiquetas y normalización
  
# Apertura del documento que contiene los datos del conjunto de 'test'
coordenadas_test = pd.read_csv("/home/laura/TrayectoriasLSTM/datasets/datasets_degradacion_v1/test/conjuntoTest_numTrayectoriasTest225_Nexus_numTrayectoriasTrain9_difGPSfreqcon0_numTrayectoriasValid2.csv",header=None) 
txt_test = 'conjuntoTest_numTrayectoriasTest225_Nexus_numTrayectoriasTrain9_difGPSfreqcon0_numTrayectoriasValid2.csv'
numeros_nombre_test = [float(s) for s in re.findall(r'-?\d+\.?\d*', txt_test)]
numTrayects_test = int(numeros_nombre_test[0])

# Estas dos sentencias siguientes no son realmente necesarias ya que no se ha extraido el conjunto de test del conjunto de entrenamiento sino que es un conjunto propio
# Lo que hacen realmente es reinicial a '0' la cuenta perteneciente a la columna de 'Index' de un tipo de dato 'Series' de Pandas, ya que cuando se divide el conjunto de entrenamiento
# y se saca el de validación de este, la columna 'Index' contiene el índice que tenía dentro de entrenamiento, a pesar de ser un dato distinto, y para crear correctamente el 
# 'Sampler' que se explica a continuación, se requiere que esta columna comience su cuenta en '0'.
cero_to_len_coordenadas_test = pd.Series(range(0,len(coordenadas_test)))
coordenadas_test = coordenadas_test.set_index([cero_to_len_coordenadas_test])

# trayectorias no cojan posiciones de trayectorias distintas. Es decir, todas las posiciones de una secuencia solo pertenecen a una misma trayectoria.
# En este dataset las nulas no están puestas a -200, por eso se ven muchísimos 1's en vez de 0's
listado_test = coordenadas_test[coordenadas_test[0]==0].index.tolist()
listado_test = np.asarray(listado_test);

coordenadas_nptest = np.asarray(coordenadas_test)

# Separación de los datos de entrada de sus etiquetas correspondientes (o 'target')
trayect_data_test, trayect_labels_test = separate_data(coordenadas_test)
# Indica el número total de posiciones, calculado como la suma de las posiciones de todas las trayectorias
numpositionstest = trayect_labels_test.shape[0]

# Normalización de los datos: se usan los valores máximos y mínimos de los datos de entrenamiento para normalizar los datos.
trayect_data_test_normalized = min_max_norm_test(trayect_data_test, minvalue, maxvalue)    

# Normalización de las coordenadas (datos objetivo o 'targets'), devolviendo los valores máximos y mínimos para la normalización 
# posterior de los datos de validación y test con respecto a los datos de entrenamiento. Al igual que se normalizan los datos para 
# su entrenamiento y prueba, se deben normalizar las etiquetas que para este caso serán la latitud y la longitud de las coordenadas 
# que se pretenden predecir.
trayect_labels_test_norm = np.zeros(trayect_labels_test.shape)
trayect_labels_test_norm[:,0], trayect_labels_test_norm[:,1] = coordenates_norm_test(trayect_labels_test, minmaxlat_train, minmaxlon_train)

# Conversión de los datos en tensores
trayect_data_test_normalized = torch.tensor(trayect_data_test_normalized)#[:,None,:]
trayect_labels_test_norm = torch.tensor(trayect_labels_test_norm, dtype= torch.float32)#[:,None,:]

# In[4]: Creación del conjunto de datos muestreador de trayectorias 

class indexSampler(Sampler):
    # https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html#Custom-Sampler
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#Sampler
    def __init__(self, listado, numpostot):
        self.lista_indices = list(listado)
        self.listado = listado
        self.indice_final = numpostot-1
        
        
    def __iter__(self):
        return iter(self.lista_indices)
    
    def __len__(self,posicion):
        ind = self.lista_indices.index(posicion)
        if ind == len(self.lista_indices) -1:
            return self.indice_final-self.listado[ind]+1
        else:
            return self.listado[ind+1]-self.listado[ind]

# Creación de los distintos muestreadores para los distintos conjuntos de datos.
testSampler = indexSampler(listado_test,numpositionstest);

# In[5]:  Creación de los dataset donde la primera dimension de los datos es el número total de trayectorias, la segunda la longitud de la trayectoria y la tercera el número de puntos de acceso.

cols = trayect_data_train_normalized.shape[1] # Representa el número de puntos de acceso que están proporcionando información.
cols_target = trayect_labels_train_norm.shape[1] # Representa las dos columnas asociadas a longitud y latitud de las coordenadas.    
    
sequence_length_test = 16 # config.sequence_length # 24
solape_test = 0

  ##################################  TEST  ##############################################
  
# La variable numsecuencias_tot_test indica cuanta secuencias de sequence_length posiciones se han creado a partir de las trayectorias que componen
# el conjunto de test, mientas que numsecuencias_cadatrayect_test es un 'array' que contiene cuantas secuencias de sequence_length
# posiciones se han creado por cada trayectoria. Esta segunda depede del número de posiciones que componga cada trayectoria. La suma de todos los 
# valones de numsecuencias_cadatrayect_test debe coincidir con numsecuencias_tot_test.
numsecuencias_tot_test, numsecuencias_cadatrayect_test = num_seq_tot(testSampler, sequence_length_test, trayect_data_test_normalized, solape_test)

# Creación del array que va a contener todas las secuencias creadas y el array que contendár su objetivo asociado.
trayectoria_ensecuencia_test = torch.empty(size=(numsecuencias_tot_test, sequence_length_test, cols))
targets_ensecuencia_test = torch.empty(size=(numsecuencias_tot_test, sequence_length_test, cols_target))

# Almacenar en los distintos arrays las secuencias y objetivos. El tamaño final será [num_secuencias_totales, longitud_secuencia, num_aps]
# para el 'array' de datos y de [num_secuencias_totales, longitud_secuencia, 2 (latitud y longitud)]
trayectoria_ensecuencia_test, targets_ensecuencia_test = completeseq_consolape(trayectoria_ensecuencia_test, targets_ensecuencia_test, 
                                                                              testSampler, trayect_data_test_normalized, 
                                                                              sequence_length_test, trayect_labels_test_norm, solape_test)

test_set  = TensorDataset(trayectoria_ensecuencia_test,  targets_ensecuencia_test)

# In[6]:  Creación de los dataloaders

tbatch_size = 1

test_loader  = DataLoader(test_set, batch_size=tbatch_size, shuffle = False, drop_last=True)  

 ############################### Comprobación funcionamiento #############################
 
for step, (data, targets) in enumerate(test_loader):
  print(data)
  print(data.shape)
  print(targets)
  print(targets.shape)
  break

# In[7]:  Creación de la arquitectura del modelo (WifiLSTM_withLinear = LSTM con Cells y una fully connected; Wifi_fullLSTMCells = LSTM hecha solo con cells)


sequence_length_train = 16 
lr = 0.005637669307211288
teacher_forcing_ratio = 0.046792982444183215
num_lay = 2
hidd_d = 116 
batch_size = 47
factor_solape = 0.24941425634282657
optimizador = 1
stepsize = 13
gamma = 0.2609678219930688
sche = False
if sche == True:
    modelname = 'model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'schedulerLRstepsize'+str(stepsize)+'gamma'+str(gamma)
else:
    modelname = 'model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'NoScheduler'
# modelname = 'models/model_numTrayectTrain187seqtrain6_numTrayectValid46_lr0.004302485697165382_TeacherForcing0.4831502728497575_numLayers2_hiddNeurons118_bs61_solape0.3080681068259487_optim1schedulerLRstepsize52gamma0.07958760930423393'


# Establecimiento del tamaño de entrada de la red = Nº de puntos de acceso vistos. Establecimiento de parámetrros de la red como
# tamaño de salida, número de capas (solo válido si se usa el modelo de WifiLSTM cuya arquitectura usa el módulo nn.LSTM en vez de 
# nn.LSTMCell) y número de nodos de cada capa oculta.
in_size = trayect_data_train_normalized.shape[1] 
out_dim = 2

# Esta clase define una arquitectura donde se tienen dos capas de LSTM definidas de manera manual usando LSTMCell. Tras eso se le aplica
# una capa 'fully_connected' para transformar el tamaño de salida de 'hidd_d' a 'out_dim'.
class WifiLSTM_withLinear(nn.Module):
    def __init__(self, i_size, hidden_size, num_classes):
        super(WifiLSTM_withLinear, self).__init__()
        self.input_size = i_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.wifi_lstm_lay1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)#, num_layers=self.num_layers,
                             #   batch_first=True)
        self.wifi_lstm_lay2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.out = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)

    def forward(self, x_in, h0,c0, h1,c1):
        (hid_lay1, c_lay1) = self.wifi_lstm_lay1(x_in, (h0,c0))
        (hid_lay2, c_lay2) = self.wifi_lstm_lay2(hid_lay1,(h1,c1))
        # r_out, (hid, c) = self.wifi_lstm(x_in, (h0,c0))
        # out = self.out(r_out)
        out = self.out(hid_lay2)
        return out, (hid_lay1, c_lay1), (hid_lay2, c_lay2)

# Esta clase define una arquitectura donde se tienen dos capas de LSTM definidas de manera manual usando LSTMCell. En este caso no se 
# aplica una 'fully_conneccted' a la salida sino que se usa otra LSTMCell para reducir la dimensionalidad a la de la salida. 
class Wifi_fullLSTMCells(nn.Module):
    def __init__(self, i_size, hidden_size, num_classes):
        super(Wifi_fullLSTMCells, self).__init__()
        self.input_size = i_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.wifi_lstm_lay1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)#, num_layers=self.num_layers,
                             #   batch_first=True)
        self.wifi_lstm_lay2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.lstm_out = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.num_classes)

    def forward(self, x_in, h0,c0, h1,c1, h2,c2):
        (hid_lay1, c_lay1) = self.wifi_lstm_lay1(x_in, (h0,c0))
        (hid_lay2, c_lay2) = self.wifi_lstm_lay2(hid_lay1,(h1,c1))
        # r_out, (hid, c) = self.wifi_lstm(x_in, (h0,c0))
        # out = self.out(r_out)
        (hid_out, c_out) = self.lstm_out(hid_lay2,(h2,c2))
        return (hid_out,c_out), (hid_lay1, c_lay1), (hid_lay2, c_lay2)


# Esta clase define una arquitectura donde se tienen dos capas de LSTM definidas de manera manual usando LSTMCell. En este caso no se 
# aplica una 'fully_conneccted' a la salida sino que se usa otra LSTMCell para reducir la dimensionalidad a la de la salida. 
class Wifi_fullLSTMCells2Lay(nn.Module):
    def __init__(self, i_size, hidden_size, num_classes):
        super(Wifi_fullLSTMCells2Lay, self).__init__()
        self.input_size = i_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.wifi_lstm_lay1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)#, num_layers=self.num_layers,
                             #   batch_first=True)
        self.wifi_lstm_out = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.num_classes)

    def forward(self, x_in, h0,c0, h1,c1):
        (hid_lay1, c_lay1) = self.wifi_lstm_lay1(x_in, (h0,c0))
        (hid_out, c_out) = self.wifi_lstm_out(hid_lay1,(h1,c1))
        return (hid_out,c_out), (hid_lay1, c_lay1)


# Declaración para el uso de la arquitectura que acaba con una capa 'fully_connected':
# lstm = WifiLSTM_withLinear(in_size,hidd_d, out_dim).to(device)
# Declaración para el uso de la arquitectura que acaba con una capa 'LSTMCell':
if num_lay == 3:
    lstm = Wifi_fullLSTMCells(in_size,hidd_d, out_dim).to(device)
elif num_lay == 2:
    lstm = Wifi_fullLSTMCells2Lay(in_size,hidd_d, out_dim).to(device)
else:
    print('Error al declarar la arquitectura')
    exit()
    
# Creación del modelo:
model = lstm
if sche == True:
    model.load_state_dict(torch.load('models/'+modelname+'.pt'))
else:
    model.load_state_dict(torch.load('models/no_scheduler/'+modelname+'.pt'))

# In[8]: Determinación del título de las imágenes para distinguir entre las trayectorias de un día y otras

archivo_trayectoriasxdia = pd.read_csv("/home/laura/TrayectoriasLSTM/datasets/datasets_degradacion_v1/test/numTrayectoriasTest225_Nexus_numTrayectoriasTrain9_numTrayectoriasValid2_numerodetrayectoriaspordia_test.csv",header=None) 
archivo_trayectoriasxdia = np.asarray(archivo_trayectoriasxdia).reshape(-1)

num_dias_que_se_midieron_trayectorias_test = len(archivo_trayectoriasxdia)

lista_dias_de_las_secuencias = []

if sum(archivo_trayectoriasxdia) == len(numsecuencias_cadatrayect_test):
    for dia in range(num_dias_que_se_midieron_trayectorias_test):
        trayectorias_xdia = archivo_trayectoriasxdia[dia]
     #   secuencias_xdia = sum(numsecuencias_cadatrayect_test[trayectorias_xdia*dia:trayectorias_xdia*(dia+1)])
        secuencias_xdia = sum(numsecuencias_cadatrayect_test[np.sum(archivo_trayectoriasxdia[0:dia]):np.sum(archivo_trayectoriasxdia[0:dia]) +trayectorias_xdia])
        print(np.sum(archivo_trayectoriasxdia[0:dia]))
        print(np.sum(archivo_trayectoriasxdia[0:dia]) +trayectorias_xdia)
        print(numsecuencias_cadatrayect_test[np.sum(archivo_trayectoriasxdia[0:dia]):np.sum(archivo_trayectoriasxdia[0:dia]) +trayectorias_xdia])
        for j in range(secuencias_xdia):
            lista_dias_de_las_secuencias.append("Trayectoria del día "+str(dia))
else:
    print('Error con el numero de secuencias de test')
    exit()
    

# In[9]: Definición de check_accuracy_imgs_seq():
    
import os
from numpy import savetxt
    
# Definición de la función que comprueba la precisión mediante una representación gráfica de las posiciones en el mapa. Además permite la
# exportación de los datos para un posterior cálculo en MATLAB del error medio.
def check_accuracy_imgs_seq(loader, net, dataset, name, loss_function, h_state, seq_len, col, test_batch_size, device, minmax, nameprop, listado_titulo): #, num_classes, cols):
    print("Checking accuracy on test data")
    t1_rep = 12
    t2_rep = 20
    t3_rep = 27
    # Creación de la carpeta donde se van a almacenar los datos.
    namedir = '../results/'+dataset+'/'+name+'/'+nameprop+'/export'
    namedir_imgs = '../results/'+dataset+'/'+name
    os.makedirs(namedir, exist_ok=True) 
    os.makedirs(namedir_imgs, exist_ok=True) 
    # namedir_props = namedir_imgs+'/'+nameprop
    # os.makedirs(namedir_props, exist_ok=True) 
    
    pred_coordenatestot, coordenatestot = [], []
    net.eval()
    minmaxlat_train = minmax[0]
    minmaxlon_train = minmax[1]

    with torch.no_grad():
        test_loss = []
        test_losses = []
        total = []
        medidas_dias = []
        var = 0
        for step, (data, target) in enumerate(loader):
            c0 = torch.zeros(tbatch_size, hidd_d).to(device)
            hidden_0 = torch.zeros(tbatch_size, hidd_d).to(device)
            c1 = torch.zeros(tbatch_size, hidd_d).to(device)
            hidden_1 = torch.zeros(tbatch_size, hidd_d).to(device)
            c2 = torch.zeros(tbatch_size, out_dim).to(device)
            hidden_2 = torch.zeros(tbatch_size, out_dim).to(device)         
            data = data.to(device=device)
            target = target.to(device=device)
            pred_coordenates, coordenates = [], []
                       
            # Paso de las secuencias de prueba por la red una vez ya entrenada:
            scores = torch.zeros(target.shape).to(device)
            for i in range(target.shape[1]):
                (hidden_2,c2), (hidden_0, c0), (hidden_1, c1) = net(data[:,i], hidden_0, c0, hidden_1, c1, hidden_2, c2) 
                scores[:,i,:] = hidden_2
            # scores, _ = net(data)
            loss = loss_function(scores, target)
            test_loss.append(loss.item())
            
            scores = scores.reshape(-1,2)
            target = target.reshape(-1,2)
            
            # Desnormalización de los datos para su representación y para su exportación para calcular el error medio en distancia:
                
            lat_denorm, lon_denorm = coordenates_denorm(target, minmaxlat_train, minmaxlon_train) # donde ~ se debe pasar la posición predicha
            coord_denorm = np.zeros((target.shape[0],2))
            coord_denorm[:,0] = lat_denorm[:].cpu().numpy()  # lat_denorm[:].numpy() # if cuda is not available = without .cpu()
            coord_denorm[:,1] = lon_denorm[:].cpu().numpy()  # lon_denorm[:].numpy() # if cuda is not available = without .cpu()

            predlat_denorm, predlon_denorm = coordenates_denorm(scores, minmaxlat_train, minmaxlon_train) # donde ~ se debe pasar la posición predicha
            predcoord_denorm = np.zeros((scores.shape[0],2))
            predcoord_denorm[:,0] = predlat_denorm[:].cpu().numpy() # predlat_denorm[:].numpy() # if cuda is not available = without .cpu()
            predcoord_denorm[:,1] = predlon_denorm[:].cpu().numpy() # predlon_denorm[:].numpy() # if cuda is not available = without .cpu()

            # Exportación de las coordenadas objetivo y de las predichas para el cálculo del error en distancia entre unas y otras:

            namedir_export_test = '../results/'+dataset+'/'+name+'/'+nameprop+'/export/Trayectoria_Dia_'+str(int(''.join(filter(str.isdigit,listado_titulo[step]))))
            os.makedirs(namedir_export_test, exist_ok=True) 
            savetxt(namedir_export_test+'/coordscores_step'+str(step)+'.csv',coord_denorm, delimiter=',')
            savetxt(namedir_export_test+'/coordtarget_step'+str(step)+'.csv',predcoord_denorm, delimiter=',') 

            if int(''.join(filter(str.isdigit,listado_titulo[step]))) != var:
                namedir_export_test_error = '../results/'+dataset+'/'+name+'/'+nameprop+'/export/Trayectoria_Dia_'+str(int(''.join(filter(str.isdigit,listado_titulo[step-1]))))
                medidas_dias = np.array(medidas_dias)
                error_medio_dia = sum(medidas_dias)/len(medidas_dias)
                max_error_dia = max(medidas_dias)
                min_error_dia = min(medidas_dias)
                mse_dia = sum(medidas_dias**2)/len(medidas_dias)
                Info = 'Para las '+str(len(medidas_dias))+' posiciones de las secuencias del dia '+str(int(''.join(filter(str.isdigit,listado_titulo[step]))))+'  del conjunto de '+dataset+', el error de las secuencias es:'
                Info = Info + '\nEl error medio es de: '+str(error_medio_dia)+' metros. \nEl error máximo es de: '+str(max_error_dia)+' metros y el mínimo es de: '+str(min_error_dia)+' metros.\n El error cuadrático medio es de: '+str(mse_dia)+' metros.'
                
                f = open(namedir_export_test_error+'/informacion_error_total_dia'+str(int(''.join(filter(str.isdigit,listado_titulo[step-1]))))+'.txt','wt')
                f.write(Info)
                f.close()
                medidas_dias = []
                var = int(''.join(filter(str.isdigit,listado_titulo[step])))
            else:
                var = int(''.join(filter(str.isdigit,listado_titulo[step])))

            distancia_coords = np.zeros((scores.shape[0]))
            for i in range(0,len(scores)):
                # distancia_coords[i] = hs.haversine(scores[i,:],target[i,:],unit=Unit.METERS)
                  distancia_coords[i] = hs.haversine(coord_denorm[i,:],predcoord_denorm[i,:],unit=Unit.METERS)
                  total.append(distancia_coords[i])
            error_medio = sum(distancia_coords)/len(distancia_coords)
            max_error = max(distancia_coords)
            min_error = min(distancia_coords)
            mse = sum(distancia_coords**2)/len(distancia_coords)
            Info = 'Para la secuencia '+str(step)+' de la trayectoria del día del conjunto de '+dataset+', el error entre cada una de las posiciones es:\n'+str(distancia_coords)+' metros.'
            Info = Info + '\nEl error medio es de: '+str(error_medio)+' metros. \nEl error máximo es de: '+str(max_error)+' metros y el mínimo es de: '+str(min_error)+' metros.\n El error cuadrático medio es de: '+str(mse)+' metros.'
            
            f = open(namedir+'/informacion_error_secuencia'+str(step)+'.txt','wt')
            f.write(Info)
            f.close()

            print('coord_target = {}  -> predicted coord = {},'.format(coord_denorm, predcoord_denorm))
            pred_coordenates.append(predcoord_denorm)
            coordenates.append(coord_denorm)
            pred_coordenatestot.append(predcoord_denorm)
            coordenatestot.append(coord_denorm)
                
            print(step)
            coordenadas_obj = np.asarray(coordenates)
            coordenadas_obj = coordenadas_obj.reshape(coordenadas_obj.shape[1],-1)
            coordenadas_pred = np.asarray(pred_coordenates)
            coordenadas_pred = coordenadas_pred.reshape(coordenadas_pred.shape[1],-1) 
            
            
            if (int(''.join(filter(str.isdigit,listado_titulo[step]))) == t1_rep) or (int(''.join(filter(str.isdigit,listado_titulo[step]))) == t2_rep) or (int(''.join(filter(str.isdigit,listado_titulo[step]))) == t3_rep):
                # Representación en el mapa de las coordenadas objetivo y las predichas:
                BBox = (-3.15210, -3.14706, 40.64149, 40.64465)      
                plt.figure()
                map_image = plt.imread('../imgs/map_GuadaAfueras2.png')
                fig, ax = plt.subplots(figsize = (8,7))
                ax.scatter(coordenadas_obj[:,1],coordenadas_obj[:,0], zorder=1, alpha= 0.6, c='b', s=10)
                ax.scatter(coordenadas_pred[:,1],coordenadas_pred[:,0], zorder=1, alpha= 0.7, c='r', s=10)                
                ax.set_title(listado_titulo[step])
                ax.set_xlim(BBox[0],BBox[1]) # EJE DE LONGITUD
                ax.set_ylim(BBox[2],BBox[3]) # EJE DE LATITUD
                ax.imshow(map_image, zorder=0, extent = BBox, aspect= 'equal') #, dpi=600)
                plt.savefig(namedir_export_test+'/Pruebamap_tray'+str(step)+'_predGuadaAfuerasTrain.png',format='png', dpi=600)
                t1_rep = t1_rep-1
                t2_rep = t2_rep-1
                t3_rep = t3_rep-1

        if int(''.join(filter(str.isdigit,listado_titulo[step]))) != var:
            namedir_export_test_error = '../results/'+dataset+'/'+name+'/'+nameprop+'/export/Trayectoria_Dia_'+str(int(''.join(filter(str.isdigit,listado_titulo[step-1]))))
            medidas_dias = np.array(medidas_dias)
            error_medio_dia = sum(medidas_dias)/len(medidas_dias)
            max_error_dia = max(medidas_dias)
            min_error_dia = min(medidas_dias)
            mse_dia = sum(medidas_dias**2)/len(medidas_dias)
            Info = 'Para las '+str(len(medidas_dias))+' posiciones de las secuencias del dia '+str(int(''.join(filter(str.isdigit,listado_titulo[step]))))+'  del conjunto de '+dataset+', el error de las secuencias es:'
            Info = Info + '\nEl error medio es de: '+str(error_medio_dia)+' metros. \nEl error máximo es de: '+str(max_error_dia)+' metros y el mínimo es de: '+str(min_error_dia)+' metros.\n El error cuadrático medio es de: '+str(mse_dia)+' metros.'
                
            f = open(namedir_export_test_error+'/informacion_error_total_dia'+str(int(''.join(filter(str.isdigit,listado_titulo[step-1]))))+'.txt','wt')
            f.write(Info)
            f.close()
            medidas_dias = []
            var = int(''.join(filter(str.isdigit,listado_titulo[step])))
        else:
            var = int(''.join(filter(str.isdigit,listado_titulo[step])))


        test_losses.append(sum(test_loss) / len(test_loss))
        total = np.array(total)
        error_medio = sum(total)/len(total)
        max_error = max(total)
        min_error = min(total)
        mse = sum(total**2)/len(total)
        Info = 'Cogiendo todas las secuencias que salen de las trayectorias tomadas para el conjunto de '+dataset+', se obtiene:'
        Info = Info + '\n\tEl error medio es de: '+str(error_medio)+' metros. \n\tEl error máximo es de: '+str(max_error)+' metros y el mínimo es de: '+str(min_error)+' metros.\n\tEl error cuadrático medio es de: '+str(mse)+' metros.'
        
        f = open(namedir+'/informacion_error_total_secuencias.txt','wt')
        f.write(Info)
        f.close()   
        return pred_coordenatestot, coordenatestot



#############################################################################################################################################################


# Definición de la función que comprueba la precisión mediante una representación gráfica de las posiciones en el mapa. Además permite la
# exportación de los datos para un posterior cálculo en MATLAB del error medio, en este caso para la arquitectura de 2 layers:
def check_accuracy_imgs_seq2lay(loader, net, dataset, name, loss_function, h_state, seq_len, col, test_batch_size, device, minmax, nameprop, listado_titulo): #, num_classes, cols):

    print("Checking accuracy on test data")
    t1_rep = 12
    t2_rep = 20
    t3_rep = 27
    # Creación de la carpeta donde se van a almacenar los datos.
    namedir = '../results/'+dataset+'/'+name+'/'+nameprop+'/export'
    namedir_imgs = '../results/'+dataset+'/'+name
    os.makedirs(namedir, exist_ok=True) 
    os.makedirs(namedir_imgs, exist_ok=True) 
    # namedir_props = namedir_imgs+'/'+nameprop
    # os.makedirs(namedir_props, exist_ok=True) 
    
    pred_coordenatestot, coordenatestot = [], []
    net.eval()
    minmaxlat_train = minmax[0]
    minmaxlon_train = minmax[1]

    with torch.no_grad():
        test_loss = []
        test_losses = []
        total = []
        medidas_dias = []
        var = 0
        for step, (data, target) in enumerate(loader):
            c0 = torch.zeros(tbatch_size, hidd_d).to(device)
            hidden_0 = torch.zeros(tbatch_size, hidd_d).to(device)
            c1 = torch.zeros(tbatch_size, out_dim).to(device)
            hidden_1 = torch.zeros(tbatch_size, out_dim).to(device)
            data = data.to(device=device)
            target = target.to(device=device)
            pred_coordenates, coordenates = [], []
            
            # Paso de las secuencias de prueba por la red una vez ya entrenada:
            scores = torch.zeros(target.shape).to(device)
            for i in range(target.shape[1]):
                (hidden_1, c1), (hidden_0, c0) = net(data[:,i], hidden_0, c0, hidden_1, c1) 
                scores[:,i,:] = hidden_1
            # scores, _ = net(data)
            loss = loss_function(scores, target)
            test_loss.append(loss.item())
            
            scores = scores.reshape(-1,2)
            target = target.reshape(-1,2)
            
            # Desnormalización de los datos para su representación y para su exportación para calcular el error medio en distancia:
                
            lat_denorm, lon_denorm = coordenates_denorm(target, minmaxlat_train, minmaxlon_train) # donde ~ se debe pasar la posición predicha
            coord_denorm = np.zeros((target.shape[0],2))
            coord_denorm[:,0] = lat_denorm[:].cpu().numpy()  # lat_denorm[:].numpy() # if cuda is not available = without .cpu()
            coord_denorm[:,1] = lon_denorm[:].cpu().numpy()  # lon_denorm[:].numpy() # if cuda is not available = without .cpu()

            predlat_denorm, predlon_denorm = coordenates_denorm(scores, minmaxlat_train, minmaxlon_train) # donde ~ se debe pasar la posición predicha
            predcoord_denorm = np.zeros((scores.shape[0],2))
            predcoord_denorm[:,0] = predlat_denorm[:].cpu().numpy() # predlat_denorm[:].numpy() # if cuda is not available = without .cpu()
            predcoord_denorm[:,1] = predlon_denorm[:].cpu().numpy() # predlon_denorm[:].numpy() # if cuda is not available = without .cpu()

            # Exportación de las coordenadas objetivo y de las predichas para el cálculo del error en distancia entre unas y otras:
            namedir_export_test = '../results/'+dataset+'/'+name+'/'+nameprop+'/export/Trayectoria_Dia_'+str(int(''.join(filter(str.isdigit,listado_titulo[step]))))
            os.makedirs(namedir_export_test, exist_ok=True) 
            savetxt(namedir_export_test+'/coordscores_step'+str(step)+'.csv',coord_denorm, delimiter=',')
            savetxt(namedir_export_test+'/coordtarget_step'+str(step)+'.csv',predcoord_denorm, delimiter=',') 

            
            distancia_coords = np.zeros((scores.shape[0]))
            if int(''.join(filter(str.isdigit,listado_titulo[step]))) != var:
                namedir_export_test_error = '../results/'+dataset+'/'+name+'/'+nameprop+'/export/Trayectoria_Dia_'+str(int(''.join(filter(str.isdigit,listado_titulo[step-1]))))
                medidas_dias = np.array(medidas_dias)
                error_medio_dia = sum(medidas_dias)/len(medidas_dias)
                max_error_dia = max(medidas_dias)
                min_error_dia = min(medidas_dias)
                mse_dia = sum(medidas_dias**2)/len(medidas_dias)
                Info = 'Para las '+str(len(medidas_dias))+' posiciones de las secuencias del dia '+str(int(''.join(filter(str.isdigit,listado_titulo[step]))))+'  del conjunto de '+dataset+', el error de las secuencias es:'
                Info = Info + '\nEl error medio es de: '+str(error_medio_dia)+' metros. \nEl error máximo es de: '+str(max_error_dia)+' metros y el mínimo es de: '+str(min_error_dia)+' metros.\n El error cuadrático medio es de: '+str(mse_dia)+' metros.'
                
                f = open(namedir_export_test_error+'/informacion_error_total_dia'+str(int(''.join(filter(str.isdigit,listado_titulo[step-1]))))+'.txt','wt')
                f.write(Info)
                f.close()
                medidas_dias = []
                var = int(''.join(filter(str.isdigit,listado_titulo[step])))
            else:
                var = int(''.join(filter(str.isdigit,listado_titulo[step])))
            for i in range(0,len(scores)):
                # distancia_coords[i] = hs.haversine(scores[i,:],target[i,:],unit=Unit.METERS)
                  distancia_coords[i] = hs.haversine(coord_denorm[i,:],predcoord_denorm[i,:],unit=Unit.METERS)
                  medidas_dias.append(distancia_coords[i])
                  total.append(distancia_coords[i])
            error_medio = sum(distancia_coords)/len(distancia_coords)
            max_error = max(distancia_coords)
            min_error = min(distancia_coords)
            mse = sum(distancia_coords**2)/len(distancia_coords)
            Info = 'Para la secuencia '+str(step)+' de la trayectoria del día del conjunto de '+dataset+', el error entre cada una de las posiciones es:\n'+str(distancia_coords)+' metros.'
            Info = Info + '\nEl error medio es de: '+str(error_medio)+' metros. \nEl error máximo es de: '+str(max_error)+' metros y el mínimo es de: '+str(min_error)+' metros.\n El error cuadrático medio es de: '+str(mse)+' metros.'
            
            f = open(namedir_export_test+'/informacion_error_secuencia'+str(step)+'.txt','wt')
            f.write(Info)
            f.close()

            
            print('coord_target = {}  -> predicted coord = {},'.format(coord_denorm, predcoord_denorm))
            pred_coordenates.append(predcoord_denorm)
            coordenates.append(coord_denorm)
            pred_coordenatestot.append(predcoord_denorm)
            coordenatestot.append(coord_denorm)
                
            print(step)
            coordenadas_obj = np.asarray(coordenates)
            coordenadas_obj = coordenadas_obj.reshape(coordenadas_obj.shape[1],-1)
            coordenadas_pred = np.asarray(pred_coordenates)
            coordenadas_pred = coordenadas_pred.reshape(coordenadas_pred.shape[1],-1) 
            
            if (int(''.join(filter(str.isdigit,listado_titulo[step]))) == t1_rep) or (int(''.join(filter(str.isdigit,listado_titulo[step]))) == t2_rep) or (int(''.join(filter(str.isdigit,listado_titulo[step]))) == t3_rep):
                # Representación en el mapa de las coordenadas objetivo y las predichas:
                BBox = (-3.15210, -3.14706, 40.64149, 40.64465)      
                plt.figure()
                map_image = plt.imread('../imgs/map_GuadaAfueras2.png')
                fig, ax = plt.subplots(figsize = (8,7))
                ax.scatter(coordenadas_obj[:,1],coordenadas_obj[:,0], zorder=1, alpha= 0.6, c='b', s=10)
                ax.scatter(coordenadas_pred[:,1],coordenadas_pred[:,0], zorder=1, alpha= 0.7, c='r', s=10)                
                ax.set_title(listado_titulo[step])
                ax.set_xlim(BBox[0],BBox[1]) # EJE DE LONGITUD
                ax.set_ylim(BBox[2],BBox[3]) # EJE DE LATITUD
                ax.imshow(map_image, zorder=0, extent = BBox, aspect= 'equal') #, dpi=600)
                plt.savefig(namedir_export_test+'/Pruebamap_tray'+str(step)+'_predGuadaAfuerasTrain.png',format='png', dpi=600)


        namedir_export_test_error = '../results/'+dataset+'/'+name+'/'+nameprop+'/export/Trayectoria_Dia_'+str(int(''.join(filter(str.isdigit,listado_titulo[step-1]))))
        medidas_dias = np.array(medidas_dias)
        error_medio_dia = sum(medidas_dias)/len(medidas_dias)
        max_error_dia = max(medidas_dias)
        min_error_dia = min(medidas_dias)
        mse_dia = sum(medidas_dias**2)/len(medidas_dias)
        Info = 'Para las '+str(len(medidas_dias))+' posiciones de las secuencias del dia '+str(int(''.join(filter(str.isdigit,listado_titulo[step]))))+'  del conjunto de '+dataset+', el error de las secuencias es:'
        Info = Info + '\nEl error medio es de: '+str(error_medio_dia)+' metros. \nEl error máximo es de: '+str(max_error_dia)+' metros y el mínimo es de: '+str(min_error_dia)+' metros.\n El error cuadrático medio es de: '+str(mse_dia)+' metros.'
                
        f = open(namedir_export_test_error+'/informacion_error_total_dia'+str(int(''.join(filter(str.isdigit,listado_titulo[step-1]))))+'.txt','wt')
        f.write(Info)
        f.close()
        medidas_dias = []
        var = int(''.join(filter(str.isdigit,listado_titulo[step])))


        test_losses.append(sum(test_loss) / len(test_loss))
        total = np.array(total)
        error_medio = sum(total)/len(total)
        max_error = max(total)
        min_error = min(total)
        mse = sum(total**2)/len(total)
        Info = 'Cogiendo todas las secuencias que salen de las trayectorias tomadas para el conjunto de '+dataset+', se obtiene:'
        Info = Info + '\n\tEl error medio es de: '+str(error_medio)+' metros. \n\tEl error máximo es de: '+str(max_error)+' metros y el mínimo es de: '+str(min_error)+' metros.\n\tEl error cuadrático medio es de: '+str(mse)+' metros.'
        
        f = open(namedir+'/informacion_error_total_secuencias.txt','wt')
        f.write(Info)
        f.close()   
        return pred_coordenatestot, coordenatestot

# In[10]:  Pruebas de test: se debe pasar batch_size uno y no representa r en la misma todas las salidas de batches sino representar una a una las 
# salidas que se van obteniendo del modelo.

minmax = [minmaxlat_train, minmaxlon_train]
test_state = torch.zeros(num_lay, tbatch_size, hidd_d, dtype=torch.float).to(device)
loss_func = nn.MSELoss() 

if num_lay == 3:
    nameprop = modelname # 'numTrayectsTest'+str(numTrayects_test)+modelname
    # new dataset:  
    predictions, targets = check_accuracy_imgs_seq(test_loader, model, 'test','numtrayectstest'+str(numTrayects_test), loss_func, test_state,sequence_length_test,in_size,tbatch_size, device, minmax, nameprop, lista_dias_de_las_secuencias)
elif num_lay == 2:
    nameprop = modelname #'numTrayectsTest'+str(numTrayects_test)+modelname
    # new dataset:  
    predictions, targets = check_accuracy_imgs_seq2lay(test_loader, model, 'test','numtrayectstest'+str(numTrayects_test), loss_func, test_state,sequence_length_test,in_size,tbatch_size, device, minmax,nameprop, lista_dias_de_las_secuencias)
else:
    print('Fallo en la comprobación de eficiencia')
