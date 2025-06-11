import torch
from torch import nn
import torch.nn.functional as F

class alex(nn.Module):
    def __init__(self, num_class=10):
        super(alex, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # nn.Conv2d(48,128, kernel_size=3),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(128, 192, kernel_size=3),
            # nn.Conv2d(192, 192, kernel_size=3),
            # nn.Conv2d(192, 128, kernel_size=3),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Flatten(),
            # nn.Linear(128 * 3 * 3, 2048),
            # nn.Linear(2048, 1024),
            # nn.Linear(1024, 10),

        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_class),
        )

    def forward(self, x):
        # y = self.model(x)
        y = F.interpolate(
            x,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        y = self.features(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)

        return y

if __name__ == '__main__':
    # x = torch.randn(1, 3, 224, 224)
    # alexnet = alex()
    # y = alexnet(x)
    # print(y.shape)

    model = alex()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)
