import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class LandUseDataset(Dataset):

    labels = ['agricultural', 'denseresidential', 'mediumresidential', 'sparseresidential'
        ,'airplane', 'forest', 'mobilehomepark', 'storagetanks'
        ,'baseballdiamond', 'freeway', 'overpass', 'tenniscourt'
        ,'beach', 'golfcourse', 'parkinglot'
        ,'buildings', 'harbor', 'river'
        ,'chaparral', 'intersection', 'runway']

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        sample  = {'image': image, 'label': self.labels.index(label)}

        return sample