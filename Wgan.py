import numpy as np
from WNentorks import Generator,Discriminator
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
import torch
from parameters import opt

check=False
path='./Wchectpoint/'


cuda = True if torch.cuda.is_available() else False
print(cuda)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#Wloss=torch.nn.MSELoss()


# Loss function
"""
def Dloss(x:torch,y:torch):
    return -x.mean()+y.mean()

def Gloss(x:torch):
    return -x.mean()
"""
# Loss function
def Dloss(x,y):
    return -torch.mean(x)+torch.mean(y)

def Gloss(x):
    return -torch.mean(x)



# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

transform=transforms.Compose([transforms.ToTensor()])


# Configure data loader
train_dataset=datasets.MNIST('./MNIST',download=True,train=True,transform=transform)

dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.glr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.dlr)

#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

start_epoch=0
if check:
    checkpoint=torch.load(path+'30')
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimG'])
    optimizer_D.load_state_dict(checkpoint['optimD'])
    start_epoch=checkpoint['start_epoch']+1


# ----------
#  Training
# ----------
for epoch in range(start_epoch,opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        real = torch.ones((imgs.shape[0],1),requires_grad=False).cuda()
        fake = torch.zeros((imgs.shape[0],1), requires_grad=False).cuda()
        # Configure input
        z =torch.randn((imgs.shape[0],100)).cuda()
        gen_imgs = generator(z).detach()
        imgs=imgs.cuda()
        optimizer_D.zero_grad()

        x=discriminator(imgs)
        y=discriminator(gen_imgs)
        d_loss = Dloss(x,y).cuda()
        d_loss.backward()
        optimizer_D.step()
        #参数截断
        for para in discriminator.parameters():
            para.data.clip(-0.01,0.01)

        optimizer_G.zero_grad()
        gen_imgs = generator(z)
        g_loss = Gloss(discriminator(gen_imgs)).cuda()  #  try to backward through the graph a second time!!!
        g_loss.backward()
        optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

    with open('./WlossG.txt','a') as f1:
        f1.write('%5f,'%(g_loss.item()))

    with open('./WlossD.txt','a') as f2:
        f2.write('%5f,'%(d_loss.item()))

    if epoch%3==0:
        checkpoint={
            'generator':generator.state_dict(),
            'discriminator':discriminator.state_dict(),
            'optimG':optimizer_G.state_dict(),
            'optimD':optimizer_D.state_dict(),
            'start_epoch':epoch
        }
        torch.save(checkpoint,path+'%d'%(epoch))
        save_image(gen_imgs,'./gen_images/sonw%d.png'%(epoch),nrow=16)

f1.close()
f2.close()
