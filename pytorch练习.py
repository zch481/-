import torch
from torch import nn
from torch.optim import Adam

device = torch.device("cuda"if torch.cuda.is_available()else"cpu")
# 0.准备数据
x = torch.rand([500, 1]).to(device)
y_true = x * 3 + 0.8


# 1.定义模型
class Mylinear(nn.Module):
    def __init__(self):
        # 继承父类的init
        super(Mylinear, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


# 2.实例化模型
my_linear = Mylinear().to(device)
optimizer =Adam(my_linear.parameters(),lr=0.001)
loss_fn = nn.MSELoss()

# 3.训练模型
for i in range(20000):
    y_predict = my_linear(x)
    loss = loss_fn(y_predict,y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(i%100==0):
        a =list(my_linear.parameters())
        print(loss.item(),a[0].item(),a[1].item())