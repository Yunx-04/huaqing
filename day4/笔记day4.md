# 笔记day4

## 加载数据集的函数

```jsx
class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path: str, folder_name, transform):
        self.transform = transform
        self.data_dir = os.path.dirname(txt_path)
        self.imgs_path = []
        self.labels = []
        self.folder_name = folder_name
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            img_path, label = line.split()
            label = int(label.strip())
            # img_path = os.path.join(self.data_dir, self.folder_name, img_path)
            self.labels.append(label)
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, i):
        path, label = self.imgs_path[i], self.labels[i]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
```

数据增强

```jsx
train_data = ImageTxtDataset(r"D:\Desktop\tcl\dataset\train.txt",
                             r"D:\Desktop\tcl\dataset\image2\train",
                             transforms.Compose([transforms.Resize(256),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])
                             )
```

```jsx
# 完整的模型训练套路(以CIFAR10为例)
import time
from tkinter import Image

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from day2.train.dataset import ImageTxtDataset
from day2.train.train import train_data
from model import *
from torchvision import transforms

train_data = ImageTxtDataset(r"C:\Users\hyxhyx\Desktop\Practice\Day3\dataset\train.txt",
                             r"C:\Users\hyxhyx\Desktop\Practice\Day3\dataset\Image2\train",
                             transforms.Compose([transforms.Resize(256),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])
                             )

# 准备数据集
test_data = torchvision.datasets.CIFAR10(root="../dataset",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True )

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度{train_data_size}")
print(f"测试数据集的长度{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(test_data,batch_size=64)

# 创建网络模型

chen = Chen()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 1e-2 相当于(10)^(-2)
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0 #记录训练的次数
total_test_step = 0 # 记录测试的次数
epoch = 10 # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("../logs_train")

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        outputs = chen(imgs)
        loss = loss_fn(outputs,targets)

        # 优化器优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    end_time = time.time()
    print(f"训练时间{end_time - start_time}")

    # 测试步骤（以测试数据上的正确率来评估模型）
    total_test_loss = 0.0
    # 整体正确个数
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = chen(imgs)
            # 损失
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy,total_test_step)
    total_test_step += 1

    # 保存每一轮训练模型
    torch.save(chen,f"model_save\\mod_{i}.pth")
    print("模型已保存")

writer.close()
```

### 调用GPU去训练

把我们的模型，数据，标签，使用” .cuda() “去推到GPU上

查看机器能否使用gpu

查看pytorch版本 是否下载gpu版本

```jsx
# 完整的模型训练套路(以CIFAR10为例)
import time

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from day2.train.dataset import ImageTxtDataset
from torchvision import transforms
# from model import *

# 准备数据集
train_data = ImageTxtDataset(r"C:\Users\hyxhyx\Desktop\Practice\Day3\dataset\train.txt",
                             r"C:\Users\hyxhyx\Desktop\Practice\Day3\dataset\Image2\train",
                             transforms.Compose([transforms.Resize(256),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])
                             )
# train_data = torchvision.datasets.CIFAR10(root="../dataset",
#                                          train=True,
#                                          transform=torchvision.transforms.ToTensor(),
#                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="../dataset",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True )

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度{train_data_size}")
print(f"测试数据集的长度{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(test_data,batch_size=64)

# 创建网络模型
class Chen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

chen = Chen()
if torch.cuda.is_available():
    chen = chen.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
# learning_rate = 1e-2 相当于(10)^(-2)
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0 #记录训练的次数
total_test_step = 0 # 记录测试的次数
epoch = 10 # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("../logs_train")

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = chen(imgs)
        loss = loss_fn(outputs,targets)

        # 优化器优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤（以测试数据上的正确率来评估模型）
    total_test_loss = 0.0
    # 整体正确个数
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = chen(imgs)
            # 损失
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy,total_test_step)
    total_test_step += 1

    # 保存每一轮训练模型
    torch.save(chen,f"model_save\\chen_{i}.pth")
    print("模型已保存")

writer.close()
```

# transormer学习笔记

```jsx
import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v = ViT(
        seq_len = 256,
        patch_size = 16,
        num_classes = 100,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    v.to(device)
    time_series = torch.randn(1, 3, 256).to(device)
    preds = v(time_series)
    print(preds.shape)
```