from __future__ import print_function, division
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch
from data_loader.iterator import LandUseDataset
import numpy as np

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(42)
np.random.seed(42)

data_transform = transforms.Compose([
        transforms.RandomPerspective(),
        transforms.RandomSizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])

dataset = LandUseDataset(root_dir='dataset', transform = data_transform)



fig = plt.figure()

skip = 100
for i in range(0, len(dataset), skip):
    sample = dataset[i]

    print(i, sample['image'].shape, sample['label'])

    ax = plt.subplot(1, 4, i//skip + 1)
    plt.tight_layout()
    ax.set_title('#{} is {}'.format(i, sample['label']))
    ax.axis('off')
    plt.imshow(sample['image'].permute(1, 2, 0))

    if i//skip == 3:
        plt.show()
        break