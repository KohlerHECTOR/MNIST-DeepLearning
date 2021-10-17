from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

import numpy as np
import datetime
from pretraitement import MonDataset
from autoencoder import AutoEncoder
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
LR = 1e-6
BATCHSIZE = 5
D_ENCODING = 256
PATH = Path("AUTOENCODER_MODELS/_lr_"+str(LR)+"_batchsize_"+str(BATCHSIZE)+"_Dencoding_"+str(D_ENCODING)+"_model.pch")

train_data = DataLoader(ds , shuffle=True , batch_size=BATCHSIZE)
writer = SummaryWriter("runs/autoencoder/_lr_"+str(LR)+"_batchsize_"+str(BATCHSIZE)+"_Dencoding_"+str(D_ENCODING))

ae = AutoEncoder(ds.datax.size(1),D_ENCODING)
ae = ae.to(device) #model is on gpu

optimizer = torch.optim.SGD(ae.parameters(), lr=LR)

epoch = 1
count = 0
for n_iter in range(epoch):
    for datax,_ in train_data:
        datax = datax.to(device)
        xhat = ae.forward(datax)
        loss = torch.norm(datax - ae.forward(datax)) ** 2
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


torch.save(ae.state_dict(), PATH)
