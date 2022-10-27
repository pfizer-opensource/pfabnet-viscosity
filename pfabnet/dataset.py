from torch.utils.data import Dataset
import torch

class ViscosityDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return torch.Tensor(self.X[index]), torch.Tensor([self.y[index]])

    def __len__(self):
        return len(self.y)

