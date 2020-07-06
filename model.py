import torch.nn
import torch.nn.functional as tnf

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,16,5)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(16,32,5)
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(32*5*5,120)
        self.fc2 = torch.nn.Linear(120,84)
        self.fc3=torch.nn.Linear(84,10)



    def forward(self,x):
        x=self.conv1(x)
        x=tnf.relu(x) #input 3 32 32 out 16 28 28
        x=self.pool(x)# 16 28 28 =>>> 16 14 14
        # x=tnf.relu(x)
        x=self.conv2(x)#16 14 14 32 10 10
        x=tnf.relu(x)
        x=self.pool1(x) # 32 5 5
        # x=tnf.relu(x)
        x=x.view(-1,32*5*5)
        x=tnf.relu(self.fc1(x))
        x = tnf.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch
device = torch.device("cuda:0" if
                      torch.cuda.is_available() else "cpu")
print(device)
input1 = torch.rand([32,3,32,32])
model = LeNet()
print(model)
output = model(input1)

