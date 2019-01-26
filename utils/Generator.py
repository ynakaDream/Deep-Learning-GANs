import torch.nn as nn
import torch.nn.functional as F


class GeneratorGAN(nn.Module):
    def __init__(self, input, output, nc):
        self.nc = nc
        super().__init__()
        self.fc1 = nn.Linear(input, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.2)
        self.drop4 = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop1(F.leaky_relu(self.fc1(x), negative_slope=0.2))
        x = self.drop2(F.leaky_relu(self.fc2(x), negative_slope=0.2))
        x = self.drop3(F.leaky_relu(self.fc3(x), negative_slope=0.2))
        x = self.drop4(F.leaky_relu(self.fc4(x), negative_slope=0.2))
        output = self.tanh(self.fc5(x))

        return output.view(x.size(0), self.nc, 64, 64)


class GeneratorDCGAN(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                                                   nn.BatchNorm2d(ngf * 8),
                                                   nn.ReLU(inplace=True)),
                                     nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                                                   nn.BatchNorm2d(ngf * 4),
                                                   nn.ReLU(inplace=True)),
                                     nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                                                   nn.BatchNorm2d(ngf * 2),
                                                   nn.ReLU(inplace=True)),
                                     nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                                                   nn.BatchNorm2d(ngf),
                                                   nn.ReLU(inplace=True)),
                                     nn.Sequential(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                                                   nn.Tanh())])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



