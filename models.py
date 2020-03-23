from torch import nn
from torch import ones_like, reshape
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=2, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(64, 128, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(128, 256, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),  # (b x 128 x 10 x 10)
            )

        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            #nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(128 * 10 * 10), out_features=2024),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2024, out_features=2024),
            nn.ReLU(),
            nn.Linear(in_features=2024, out_features=1),
            #nn.Sigmoid()
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[8].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)

    def forward(self, x):
        x = reshape(x, (-1, 1, 110, 110))
        x = self.net(x)
        x = x.view(-1, 128* 10 *10)  # reduce the dimensions for linear layer input
        x = self.classifier(x)
        x= x.view(-1)
        return x


def balanced_loss(outputs, target):
    target_np = target.cpu().numpy()
    n = len(target_np)
    not_present = np.sum(target_np == 0)
    present = np.sum(target_np == 1)
    ratio = 1 - present/not_present
    weight_rebal = ones_like(target) * ratio  +  (1.0 - ratio) * target

    return  nn.functional.binary_cross_entropy(outputs, target, weight_rebal)

if __name__ == '__main__':
    import torch
    import numpy as np

    a = torch.zeros(60,1,110,110)
    model = Model()
    print(model.net(a).shape)