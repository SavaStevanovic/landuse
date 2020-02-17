from __future__ import print_function, division
import torch
import numpy as np
from model.resnet import ResNet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from model_fitting.train import fit
from visualization.iterator_sample_ploter import display_iterator_sample
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
loss_writer = SummaryWriter(os.path.join('logs', log_datatime, 'loss'))
accuracy_writer = SummaryWriter(os.path.join('logs', log_datatime, 'accuracy'))

validationset = dataset_creator.get_validation_iterator()
validationloader = torch.utils.data.DataLoader(validationset, batch_size=32, shuffle=False, num_workers=0)

fit(net, trainloader, validationloader, loss_writer, accuracy_writer, trainset.labels, epochs=100)

loss_writer.close()
accuracy_writer.close()