from torch.utils.data import Dataset, DataLoader

import torch
from torch import flatten, from_numpy, tensor
from torch.nn.functional import normalize, one_hot
class MonDataset(Dataset):
    def __init__(self, x, y, moyenne, std, do_one_hot = False):
        self.datax = from_numpy(x).flatten(start_dim=1)
        self.datay = tensor(y, dtype = torch.long)
        print(self.datay[0])
        if do_one_hot:
            self.datay = one_hot(self.datay, num_classes = 10)

        self.datax = (self.datax - moyenne)/std


        """normalize between 0 and 1"""
        self.datax
    def __getitem__(self,index):
        """ retourne un couple (ensemble,label) correspondant à l'index"""
        return (self.datax[index], self.datay[index])
    def __len__(self):
        """ renvoie la taille du jeu de donneees """
        return self.datax.size()[0]
