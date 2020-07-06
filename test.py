# import torch
# print(torch.cuda.is_available())
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# np.random.seed(123)
# nd4 = np.random.rand(2,3)
# print(nd4)
# nd5 = np.full((4,4,4),70)
# cv.imwrite("xxx.jpg",nd5)
np.random.seed(100)
x = np.linspace(-1,1,100).reshape(100,1)
y = 3*np.power(x,2)+2+0.2*np.random.rand(x.size).reshape(100,1)
plt.scatter(x,y)
plt.show()

