import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, STL10


class DatasetConverter:
    def __init__(self, dataset, resize=64, batch_size=1, is_download=True):
        '''
        :param resize: resize the input size to the given size (default: '64')
        :param dataset: choose one from 'mnist', 'stl10' and 'cifar10'
        :param batch_size: how many samples per batch to load (default: `1`)
        :param is_download: if 'True', the dataset is downloaded from the internet, and is put in './dataset' directory (default: 'True')
        '''

        self.dataset = dataset
        self.is_download = is_download
        self.data_dir = './dataset'
        self.tranform = transforms.Compose(
            [transforms.Resize(resize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_set = None
        self.test_set = None
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None

    def downloader(self):
        if self.dataset == 'mnist':
            self.train_set = MNIST(self.data_dir, train=True, transform=self.tranform, download=self.is_download)
            self.test_set = MNIST(self.data_dir, train=False, transform=self.tranform, download=self.is_download)

        elif self.dataset == 'stl10':
            self.train_set = STL10(self.data_dir, split='train', transform=self.tranform, download=self.is_download)
            self.test_set = STL10(self.data_dir, split='test', transform=self.tranform, download=self.is_download)

        elif self.dataset == 'cifar10':
            self.train_set = CIFAR10(self.data_dir, train=True, transform=self.tranform, download=self.is_download)
            self.test_set = CIFAR10(self.data_dir, train=False, transform=self.tranform, download=self.is_download)

        else:
            print('{} dataset: Not Found !!'.format(self.dataset))

    def loader(self):
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def run(self):
        '''
        :return: train_loader, test_loader
        '''
        self.downloader()
        self.loader()
        return (self.train_loader, self.test_loader)
