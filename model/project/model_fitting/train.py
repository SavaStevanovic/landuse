import torch.optim as optim
import torch.nn as nn
from visualization.tensorboard_figure_ploter import plot_classes_preds

def fit(net, trainloader, writer, classes, epoch=1, logging_perriod=10):
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