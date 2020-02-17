import torch.optim as optim
import torch.nn as nn
from visualization.tensorboard_figure_ploter import plot_classes_preds
from model_fitting.evaluate import accuracy
import torch
import os

def fit_epoch(net, trainloader, writer, classes, epoch=1):
    net.train()
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss(reduction='mean')
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'].to('cuda'), data['label'].to('cuda')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ...log the running loss
    writer.add_scalar('training loss', running_loss/len(trainloader), epoch)

    # ...log a Matplotlib Figure showing the model's predictions on a
    # random mini-batch
    writer.add_figure('predictions vs. actuals', plot_classes_preds(net, inputs, labels, classes), global_step=epoch)

def fit(net, trainloader, validationloader, loss_writer, accuracy_writer, classes, epochs=1000):
    best_acc = 0
    for epoch in range(epochs):
        fit_epoch(net, trainloader, loss_writer, classes, epoch=epoch)
        train_acc = accuracy(net, trainloader, epoch)
        val_acc = accuracy(net, validationloader, epoch)
        accuracy_writer.add_scalars('Accuracy', {'train':train_acc, 'validation':val_acc}, epoch)
        if val_acc>best_acc:
            best_acc = val_acc
            print('Saving model with accuracy: {}'.format(val_acc))
            torch.save(net, os.path.join('checkpoints', 'checkpoints.pth'))
        else:
            print('Epoch {} accuracy: {}'.format(epoch, val_acc))
    print('Finished Training')