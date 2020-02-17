from __future__ import print_function, division
import pandas as pd
from torchvision import transforms, utils
import torch
from data_loader.iterator import LandUseDataset
import numpy as np
from model.resnet import ResNet
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn as nn
import os
from model_fitting.evaluate import accuracy
from model_fitting.train import fit
from visualization.iterator_sample_ploter import display_iterator_sample
from visualization.tensorboard_figure_ploter import matplotlib_imshow

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


torch.manual_seed(42)
np.random.seed(42)

data_transform = transforms.Compose([
        # transforms.RandomPerspective(),
        transforms.RandomSizedCrop(128),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(),
        transforms.ToTensor(),
        # transforms.RandomErasing(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])

trainset = LandUseDataset(root_dir='dataset', transform = data_transform)

display_iterator_sample(trainset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)

net = ResNet(22)
net.cuda()

writer = SummaryWriter(os.path.join('logs', str(datetime.now().time())))

dataiter = iter(trainloader)
data = dataiter.next()
images, labels = data['image'].to('cuda'), data['label'].to('cuda')
# create grid of images
img_grid = utils.make_grid(images)

# show images
matplotlib_imshow(img_grid.to('cpu'), one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid.to('cpu'))

writer.add_graph(net, images)



optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()

fit(net, trainloader, optimizer, criterion, writer, LandUseDataset.labels, epochs=10, logging_perriod=10)

accuracy(net, trainloader)

writer.close()