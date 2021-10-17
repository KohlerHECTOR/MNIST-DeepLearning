from pathlib import Path
import os
import torch
from torch import nn
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
ds = MonDataset(train_images, train_labels, moyenne, std, do_one_hot = True)

d_in = ds.datax.size(1)
d_out = ds.datay.size(1)
print("DIM IN: ", d_in)
print("DIM OUT: ", d_out)
classifModel = nn.Sequential(nn.Linear(d_in, 100), nn.ReLU(), nn.Linear(100,d_out ))
classifModel = classifModel.to(device)
CEL = nn.CrossEntropyLoss()
