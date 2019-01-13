import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorGAN(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.fc1 = nn.Linear(input, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, output)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.2)
        output = self.sig(self.fc5(x))

        return output.squeeze()
