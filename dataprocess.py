import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class PhysicsDataset(Dataset):
    def __init__(self, step_x=30, step_y=30, device="cpu"):
        f = lambda x: x#(torch.sin(x*np.pi-np.pi/2.0) + 1.0) / 2.0
        x = f(torch.linspace(0,1.0,step_x))
        y = f(torch.linspace(0,1.0,step_y))
        X, Y = np.meshgrid(x, y, indexing='ij')

        self.step_x = step_x
        self.step_y = step_y
        
        # تبدیل ماتریس‌های X، Y و T به تن
        self.data:torch.tensor = torch.from_numpy(np.stack((X.ravel(), Y.ravel()), axis=1)).requires_grad_(True).to(device)
        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:]
    

class LabeledDataset(Dataset):
    def __init__(self, df:pd.DataFrame, device="cpu"):
        super().__init__()
        array_data = df.astype("float32").to_numpy()
        x_cond = np.unique(array_data[:,0])[::-24]
        y_cond = np.unique(array_data[:,1])[::-24]
        array_data = array_data[np.logical_and(np.in1d(array_data[:,0], x_cond) , np.in1d(array_data[:,1], y_cond))]
        self.data = torch.from_numpy(array_data).to(device)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :2], self.data[idx, 2:]
    
