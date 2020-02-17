import os 
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader.iterator import LandUseDataset
from torchvision import transforms

class DatasetCreator:
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = self.create_dataset(root_dir)
        self.train_data, self.validation_data = train_test_split(self.data, test_size=0.2)

    def create_dataset(self, path):
            data = []
            for subdir, dirs, _ in os.walk(path):
                for dir in dirs:
                    for img in os.listdir(os.path.join(subdir, dir)):
                        data.append((os.path.join(subdir, dir, img), dir))
            return np.array(data)

    def get_train_iterator(self, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomPerspective(),
                transforms.RandomSizedCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.RandomErasing(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ])
        return LandUseDataset(self.train_data, transform)

    def get_validation_iterator(self, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ])
        return LandUseDataset(self.validation_data, transform) 