import torch

def accuracy(net, testloader, writer, epoch=1):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data['image'].to('cuda'), data['label'].to('cuda')
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    writer.add_scalar('Validation accuracy', acc, epoch)
    return acc