import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import STL10


def convert_into_batch(batch_size, is_download=False):
    '''
    Convert STL10 dataset into batch size

    :param is_download: if "True", a STL10 dataset is downloaded to a dataset directory
    :param batch_size: batch size of training and test datasets

    :return: (train_batches, test_batches, class_labels, BATCH_SIZE)
    '''

    transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(88 / 96, 1.0), ratio=(1., 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = STL10(root='./dataset', split='train+unlabeled', download=is_download, transform=transform)
    test_set = STL10(root='./dataset', split='test', download=is_download, transform=transform)

    dataset = train_set + test_set

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    with open('./dataset/stl10_binary/class_names.txt', 'rt') as f:
        class_labels = [s.strip() for s in f.readlines()]

    return (dataloader, class_labels, batch_size)


def _imshow(batch_name, class_labels, batch_size, title=None):
    img_num = np.random.choice(batch_size, 8)
    img, axis = plt.subplots(2, 4, figsize=(6, 4))
    plt.suptitle(title.capitalize() + ' Dataset Images')
    for i, (images, labels) in enumerate(batch_name):
        j = img_num[i]
        images[j] = images[j] / 2 + 0.5
        npimg = images[j].numpy()
        axis[i // 4, i % 4].imshow(npimg.reshape(3, 96, 96).transpose((1, 2, 0)))
        axis[i // 4, i % 4].set_title(class_labels[labels[j]], fontsize=10)
        axis[i // 4, i % 4].axis('off')
        if i == 7:
            break
    plt.show()


def random_imshow(show_img='train'):
    (train_batches, test_batches, class_labels, batch_size) = convert_into_batch(batch_size=50, is_download=False)
    if show_img == 'train':
        _imshow(train_batches, class_labels, batch_size, show_img)

    elif show_img == 'test':
        _imshow(test_batches, class_labels, batch_size, show_img)


if __name__ == '__main__':
    random_imshow(show_img='train')
