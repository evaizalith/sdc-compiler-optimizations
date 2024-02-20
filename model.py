from torch import nn
from torch.utils.data import DataLoder, Dataset

class sdcModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
