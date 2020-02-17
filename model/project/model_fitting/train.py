
from visualization.tensorboard_figure_ploter import plot_classes_preds

def fit(net, trainloader, optimizer, criterion, writer, classes, epochs=1, logging_perriod=10):
    running_loss = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times

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
            if i % logging_perriod == 0:    # every 10 mini-batches...

                # ...log the running loss
                writer.add_scalar('training loss',
                                running_loss / logging_perriod,
                                epoch * len(trainloader) + i)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                writer.add_figure('predictions vs. actuals',
                                plot_classes_preds(net, inputs, labels, classes),
                                global_step=epoch * len(trainloader) + i)
                running_loss = 0.0
    print('Finished Training')