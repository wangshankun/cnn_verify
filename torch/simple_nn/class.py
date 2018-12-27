# -*- coding: UTF-8 -*-
import torch
from torch import nn, optim
from torch.autograd import Variable
#from torch.utils.data import DataLoader
import torch.nn.functional as F 
import torch.utils.data as Data
from torchvision import transforms
from torchvision import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
stdsc = StandardScaler()   #StandardScaler类,利用接口在训练集上计算均值和标准差，以便于在后续的测试集上进>行相同的缩放


BATCH_SIZE = 32
learning_rate = 1e-2
num_epoches = 50

#剔除错误类别
data   = np.load("eye_np_64.npy")
target = np.load("val_np_64.npy")

ex_index = np.where(target > 63)
target = np.delete(target,ex_index,axis = 0)
data   = np.delete(data,ex_index,axis = 0)

x,y = stdsc.fit_transform(data),target   #数据归一化
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.1, random_state=0)


x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int))
x_test  = torch.from_numpy(x_test.astype(np.float32))
y_test  = torch.from_numpy(y_test.astype(np.int))

# 先转换成 torch 能识别的 Dataset
train_dataset = Data.TensorDataset(x_train, y_train)
# 把 dataset 放入 DataLoader
train_loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=4,              # 多线程来读数据
)
test_dataset = Data.TensorDataset(x_test, y_test)
# 把 dataset 放入 DataLoader
test_loader = Data.DataLoader(
    dataset=test_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=4,              # 多线程来读数据
)

# 定义简单的三层神经网络
class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        self.bn1 = nn.BatchNorm1d(n_hidden_1, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(n_hidden_2, momentum=0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(F.relu(self.bn1(x)))
        x = self.layer3(F.relu(self.bn2(x)))
        return x

#输入两个坐标，输出两个坐标
model = Neuralnetwork(68, 256, 256, 64)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
   
    for i, data in enumerate(train_loader, 1):
        img, label = data
        np_img = img.numpy()
        img = img.view(img.size(0), -1)
        np_img = img.numpy()
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        # 向前传播
        np_img = img.numpy()
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (BATCH_SIZE * i),
                running_acc / (BATCH_SIZE * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
