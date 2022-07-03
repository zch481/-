import os
import torch
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize

# 进行归一化和标准化
transform = Compose([
    ToTensor(),
    Normalize(mean=(0.1307,), std=(0.3081,))
])
# 1.MNIST数据集的训练和测试数据的加载
train_data = MNIST(root="./data", train=True, download=False, transform=transform)
train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=64)
train_test = MNIST(root="./data", train=False, download=False, transform=transform)
test_loader = DataLoader(dataset=train_test, shuffle=False, batch_size=64)


# 2.构建模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        return x


# 3.实例化模型，损失函数
model = Model()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.255)  # Adam优化器更新参数

# 4.保存更新参数
if os.path.exists(r'./model/model.pkl'):
    model.load_state_dict(torch.load(r'./model/model.pkl'))  # 加载已保存的参数
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

def train(epoch):
    for index, data in enumerate(train_loader):
        input, target = data  # 输入数据和目标值
        optimizer.zero_grad()
        y_predict = model(input)
        loss = criterion(y_predict, target)
        loss.backward()
        optimizer.step()
        if index % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")
            print(index, loss.item())


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            input, target = data
            output = model(input)
            a,predict = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (predict == target).sum().item()
        print(correct / total)


if __name__ == '__main__':
     for i in range(5):
        train(i)
        test()
