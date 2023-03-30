import torch.nn as nn
from parameters import opt

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1=nn.Conv2d(4,512,kernel_size=3,stride=2,padding=2)
        self.convt_layers=nn.Sequential(
            nn.BatchNorm2d(512),# 输入图像的通道数
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=(3,3),stride=(3,3),padding=(2,2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3, padding=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=5, stride=3, padding=8),
            #nn.BatchNorm2d(1),
            nn.Tanh(),)

    def forward(self, z):
        #z不是通过dataloader导入的,不知道也不会自动处理batch_size
        out = z.reshape(z.shape[0],4,5,5)
        out = self.conv1(out)
        img = self.convt_layers(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 512, bn=False),
            *discriminator_block(512, 256),
            *discriminator_block(256, 128),
            *discriminator_block(128, 1),
            nn.Flatten(),
            nn.Linear(4, 1)
        )
        #self.outlayer=nn.Sequential(nn.Sigmoid())gen_imgs = generator(z).detach()
    def forward(self, img):
        out = self.model(img)
        return out
