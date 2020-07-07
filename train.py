import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim


from model import LeNet
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                          shuffle=True, num_workers=0)
test_date_iter = iter(train_loader)
test_image,test_lable = test_date_iter.next()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# def imshow(img):
#     img = img/2+0.5
#     nping = img.numpy()
#     plt.imshow(np.transpose(nping,(1,2,0)))
#     plt.show()
# print(''.join('%5s' % classes[test_lable[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_image))
net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer  = optim.Adam(net.parameters(),lr=0.001)
for epoch in range(5):
    running_loss = 0.0
    for step,data in enumerate(train_loader,start=0):
        input,labels = data
        optimizer.zero_grad()
        outputs=net(input)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 500 == 499:  # print every 500 mini-batches
            with torch.no_grad():
                outputs = net(test_image)  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]
                print(test_lable.size(0))
                accuracy = (predict_y == test_lable).sum().item() / test_lable.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0
print('Finished Training')
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)
