from __future__ import print_function, division
import pandas as pd
from torchvision import utils
import torch
import numpy as np
from model.resnet import ResNet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from model_fitting.evaluate import accuracy
from model_fitting.train import fit
from visualization.iterator_sample_ploter import display_iterator_sample
from visualization.tensorboard_figure_ploter import matplotlib_imshow
from data_loader.dataset_creator import DatasetCreator

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_creator = DatasetCreator(root_dir='./dataset')
trainset = dataset_creator.get_train_iterator()
display_iterator_sample(trainset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)

net = ResNet(22)
net.cuda()

log_datatime = str(datetime.now().time())
train_writer = SummaryWriter(os.path.join('logs', log_datatime, 'train'))
validation_writer = SummaryWriter(os.path.join('logs', log_datatime, 'validation'))

validationset = dataset_creator.get_validation_iterator()
validationloader = torch.utils.data.DataLoader(validationset, batch_size=32, shuffle=False, num_workers=0)
best_acc = 0
for epoch in range(100):
    fit(net, trainloader, train_writer, trainset.labels, epoch=epoch, logging_perriod=10)
    acc = accuracy(net, validationloader, validation_writer, epoch)
    if acc>best_acc:
        best_acc = acc
        print('Saving model with accuracy: {}'.format(acc))
        torch.save(net.state_dict(), os.path.join('checkpoints', 'checkpoints-{}.pth'.format(epoch)))
    else:
        print('Epoch {} accuracy: {}'.format(epoch, acc))
print('Finished Training')

train_writer.close()
validation_writer.close()