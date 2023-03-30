import torch

from networks import Generator
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import cv2

"""
# 理想收敛值 0.6931
a=torch.tensor(0.5)
b=torch.tensor(1.0)
l=torch.nn.BCELoss()
loss=l(a,b)
print(loss)
"""


"""
G=Generator().cuda()
checkpoint= torch.load('./checkpoint/G_withweight/200')

G.load_state_dict(torch.load(checkpoint['generator']))
G.eval()
"""

G=Generator().cuda()
checkpoint= torch.load('./checkpoint/epoch50')
G.load_state_dict(checkpoint['generator'])

image_size=32
channel=3
batch=16
z = torch.randn((batch,100)).cuda()




with torch.no_grad():
    #plt.figure(figsize=(1,1))
    imgs=G(z)
    print(imgs.size())

save_image(imgs,'./images/sonw.png',nrow=4)
