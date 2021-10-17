from torch import nn
from torch import rand
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):

        super().__init__()
        print("AUTOENCODER D_IN: ",  dim_in)
        print("AUTOENCODER D_OUT: ", dim_out)
        self.d_in = dim_in
        self.d_out = dim_out

        self.lin1 = nn.Linear(self.d_in,self.d_out)
        self.relu = F.relu
        self.tanh = F.tanh
        self.linear = F.linear
        self.bias_lin2 = nn.Parameter(rand(self.d_in))

    def encode(self,datax):
        return self.relu(self.lin1.forward(datax))

    def decode(self,encode_out):
        return self.tanh(self.linear(encode_out, self.lin1.weight.T, self.bias_lin2))

    def forward(self,datax):
        return self.decode(self.encode(datax))
