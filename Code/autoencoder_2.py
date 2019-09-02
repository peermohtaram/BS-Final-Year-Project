#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:00:45 2019

@author: matlabclient01
"""

import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Loading and Transforming data

print('upto 128 with weight_decay=1e-7 kernel = 3')

os.mkdir('./encoded_images_test')
os.mkdir('./encoded_images_test/train')
os.mkdir('./encoded_images_test/train/mitosis')
os.mkdir('./encoded_images_test/train/nonmitosis')
os.mkdir('./encoded_images_test/val')
os.mkdir('./encoded_images_test/val/mitosis')
os.mkdir('./encoded_images_test/val/nonmitosis')

transform = transforms.Compose([
        
                                transforms.ToTensor(),
#                                transforms.RandomHorizontalFlip(),
#                                transforms.RandomVerticalFlip(),
#                                transforms.RandomRotation(degrees = 90, resample=False, expand=False, center=None)
                                ]) #, transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1))])

trainset = tv.datasets.ImageFolder(root='/home/matlabclient01/Peaky_Blinders_2/test_data', transform = transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=0)

#valset = tv.datasets.ImageFolder(root='/home/matlabclient01/Peaky_Blinders_2/dataset/val/',
#                                   transform = transform)
#val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=0)

files = []
for i in range(0, len(trainset)):
    splits = trainset.samples[i][0].split('/')
    splits = splits[-1].split('.') 
    files.append(splits[0])
    
#files_2 = []
#for i in range(0, len(valset)):
#    splits_2 = valset.samples[i][0].split('/')
#    splits_2 = splits_2[-1].split('.') 
#    files_2.append(splits_2[0])

# Writing our model

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(8,16,kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(16,32,kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(32,64,kernel_size=3),
            nn.ReLU(True),
#            nn.Conv2d(64,128,kernel_size=3),
#            nn.ReLU(True),
            nn.Conv2d(64,3,kernel_size=3),
            nn.ReLU(True))

        self.decoder = nn.Sequential(             
#            nn.ConvTranspose2d(3,128,kernel_size=3),
#            nn.ReLU(True),
            nn.ConvTranspose2d(3,64,kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32,kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,16,kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(8,3,kernel_size=3),
            nn.ReLU(True),
            )

    def forward(self,x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x1, x2
    
#defining some params

num_epochs = 50
#batch_size = 128

model = Autoencoder().cuda()

distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters() ,weight_decay=1e-5,)

k=0
j = 0
len(train_loader.dataset)
t_loss = []
v_loss = []

for epoch in range(0, num_epochs):
    running_t_loss= 0
    running_v_loss= 0
    for data in train_loader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        enc, dec = model(img)
        loss = distance(dec, img)
        
        running_t_loss += loss.item() * img.size(0)
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
        
        if epoch == (num_epochs-1):
            for i in range(0,len(data[0])):
                if data[1][i] == 0:
                    save_image(enc[i], './encoded_images_test/train/mitosis/{}.jpg'.format(files[k]))
                else:
                    save_image(enc[i], './encoded_images_test/train/nonmitosis/{}.jpg'.format(files[k]))
                k +=1
    epoch_t_loss = running_t_loss / len(trainset)
    
#    with torch.no_grad():
#        for data_2 in val_loader:
#            img_2, _ = data_2
#            img_2 = Variable(img_2).cuda()
#            
#            enc_2, dec_2 = model(img_2)
#            val_loss = distance(dec_2,img_2)
#            
#            running_v_loss += val_loss.item() * img_2.size(0)
#            
##            if epoch == (num_epochs-1):
##                for i in range(0,len(data_2[0])):
##                    if data_2[1][i] == 0:
##                        save_image(enc_2[i], './encoded_images_5/val/mitosis/{}.jpg'.format(files_2[j]))
##                    else:
##                        save_image(enc_2[i], './encoded_images_5/val/nonmitosis/{}.jpg'.format(files_2[j]))
##                    j +=1
#            
#        
#        epoch_v_loss = running_v_loss / len(valset)
#    # ===================log========================
#    t_loss.append(epoch_t_loss)
#    v_loss.append(epoch_v_loss)
    print('epoch [{}/{}], training_loss:{:.4f}'.format(epoch+1, num_epochs, epoch_t_loss))

#print('with weight_decay=1e-3 kernel = 7')
plt.plot(t_loss)
plt.plot(v_loss)
plt.legend(('Train Loss'), loc='upper right')
plt.title('Autoencoder Training Plot')
plt.show()

torch.save(model.encoder, './saved_models/encoder_test.pth')
