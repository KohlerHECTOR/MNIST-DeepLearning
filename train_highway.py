from highway_network import Highway
from pathlib import Path
import os
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

import numpy as np
import datetime
from pretraitement import MonDataset
from torch.utils.tensorboard import SummaryWriter
# Téléchargement des données
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()
moyenne = train_images.mean()
std = train_images.std()
ds = MonDataset(train_images, train_labels, moyenne, std)
#  TODO:

d_in = ds.datax.size(1)
d_out = 10
d_hidden = d_in
LR = 1e-8
BATCHSIZE = 5
D_ENCODING = d_in
ARCHI = "HIGHWAY/" # AUTOENCODER, HIGHWAY
PATH = Path("CLASSIF_MODELS/"+ARCHI+"_Dhidden_"+str(d_hidden)+"_lr_"+str(LR)+"_batchsize_"+str(BATCHSIZE)+"_Dencoding_"+str(D_ENCODING)+"_model.pch")

print("DIM IN: ", d_in)
print("DIM OUT: ", d_out)

train_data = DataLoader(ds , shuffle=True , batch_size=BATCHSIZE)
writer = SummaryWriter("runs/classif/"+str(ARCHI)+"_Dhidden_"+str(d_hidden)+"_lr_"+str(LR)+"_batchsize_"+str(BATCHSIZE)+"_Dencoding_"+str(D_ENCODING))

classifModel = nn.Sequential(Highway(d_in, d_in),
                             Highway(d_in, d_in),
                             Highway(d_in, d_in),
                             Highway(d_in, d_in),
                             Highway(d_in, d_in),
                             Highway(d_in, d_in),
                             Highway(d_in, d_in),
                             nn.Linear(d_in,d_out))
classifModel = classifModel.to(device)
CEL = nn.CrossEntropyLoss()
CEL = CEL.to(device)
optimizer = torch.optim.SGD(classifModel.parameters(), lr=LR)

epoch = 10
count = 0
for n_iter in range(epoch):
    for datax,datay in train_data:
        datax = datax.to(device)
        datay = datay.to(device)
        yhat = classifModel.forward(datax)
        loss = CEL.forward(yhat, datay)
        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', loss, count)
        #
        # # Sortie directe
        print(f"Itérations {count}: loss {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1


torch.save(classifModel.state_dict(), PATH)

##TEST
moyenne = test_images.mean()
std = test_images.std()
ds_test = MonDataset(test_images, test_labels, moyenne, std)
ds_test.datax = ds_test.datax.to(device)
ds_test.datay = ds_test.datay.to(device)
print(ds_test.datay[0])
test_predictions = classifModel.forward(ds_test.datax)
test_predictions = torch.argmax(test_predictions, dim = 1)
print(test_predictions[0])
score = 0
for i in range(len(test_predictions)):
    if test_predictions[i] == ds_test.datay[i]:
        score +=1

score /= len(test_predictions)
print("TEST SCORE = ", score)
