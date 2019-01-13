import torch.nn as nn


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
