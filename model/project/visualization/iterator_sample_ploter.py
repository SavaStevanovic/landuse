import matplotlib.pyplot as plt


def display_iterator_sample(trainset, skip=100):
    fig = plt.figure()

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