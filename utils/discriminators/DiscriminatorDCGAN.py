import torch.nn as nn


class DiscriminatorDCGAN(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                                                   nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                                     nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                                                   nn.BatchNorm2d(ndf * 2),
                                                   nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                                     nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                                                   nn.BatchNorm2d(ndf * 4),
                                                   nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                                     nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                                                   nn.BatchNorm2d(ndf * 8),
                                                   nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                                     nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 1, 0),
                                                   nn.Sigmoid())])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()
