from __future__ import print_function, division
import pandas as pd
import matplotlib.pyplot as plt
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
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img + 1.     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            LandUseDataset.labels[preds[idx]],
            probs[idx] * 100.0,
            LandUseDataset.labels[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

torch.manual_seed(42)
np.random.seed(42)

data_transform = transforms.Compose([
        transforms.RandomPerspective(),
        transforms.RandomSizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])

trainset = LandUseDataset(root_dir='dataset', transform = data_transform)



fig = plt.figure()

skip = 100
for i in range(0, len(trainset), skip):
    sample = trainset[i]

    print(i, sample['image'].shape, sample['label'])

    ax = plt.subplot(1, 4, i//skip + 1)
    plt.tight_layout()
    ax.set_title('#{} is {}'.format(i, trainset.labels[sample['label']]))
    ax.axis('off')
    plt.imshow(sample['image'].permute(1, 2, 0))

    if i//skip == 3:
        plt.show()
        break

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=10)

criterion = nn.CrossEntropyLoss()
net = ResNet(22)
optimizer = optim.Adam(net.parameters())

writer = SummaryWriter(os.path.join('logs', str(datetime.now().time())))

dataiter = iter(trainloader)
data = dataiter.next()
images, labels = data['image'], data['label']
# create grid of images
img_grid = utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

writer.add_graph(net, images)
writer.close()



running_loss = 0.0
for epoch in range(10):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'], data['label']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:    # every 10 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 10,
                            epoch * len(trainloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Training')