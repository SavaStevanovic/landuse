import torch

def accuracy(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data['image'].to('cuda'), data['label'].to('cuda')
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            try:
                correct += (predicted == labels).sum().item()
            except expression as identifier:
                pass


    print('Accuracy of the network: %d %%' % (
        100 * correct / total))