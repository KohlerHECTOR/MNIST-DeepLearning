from torch import nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.lin = nn.Linear(self.d_in,self.d_out)
        self.sigm = F.sigmoid

    def forward(self, x):
        H = self.lin.forward(x)
        T = self.sigm(H)
        # print(H.size(), T.size(), x.size())
        return H * T + x * (1 - T)


highway = Highway(10,32)
