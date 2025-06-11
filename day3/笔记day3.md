# 笔记day3

搭建alexnet

```jsx
import torch
from torch import nn

class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=4),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(48, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 192, kernel_size=3),
            nn.Conv2d(192, 192, kernel_size=3),
            nn.Conv2d(192, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        y = self.model(x)

        return y

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    alexnet = alex()
    y = alexnet(x)
    print(y.shape)
```

### 激活函数

```jsx
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入数据集
dataset = torchvision.datasets.CIFAR10(root="dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)

# 设置input
input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

# 非线性激活网络
class Chen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

chen = Chen()

writer = SummaryWriter("sigmod_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output_sigmod = chen(imgs)
    writer.add_images("output", output_sigmod, global_step=step)
    step += 1
writer.close()

output = chen(input)
print(output)

```

### 尝试使用resnet，Googlenet，mobileNet，moganet 等不同模型跑图片分类

### 尝试使用GPU训练网络模型