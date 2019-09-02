#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:21:38 2019

@author: matlabclient01
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 02:12:28 2019

@author: Abdullah
"""
import matplotlib.pyplot as plt
import cv2 as cv2
from skimage.feature import local_binary_pattern
import numpy as np
import glob
import torch
from torchvision import transforms

import pywt
import pywt.data
from skimage.feature import hog
from skimage.transform import resize
from sklearn.metrics import f1_score
#astype('float32')

lbp_val=[]
lbp_train=[]
METHOD='uniform'
path2 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/dataset/train/nonmitosis/*.jpg')
path3 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/dataset/train/mitosis/*.jpg')
path4 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/dataset/val/nonmitosis/*.jpg')
path5 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/dataset/val/mitosis/*.jpg')
net_PATH='/home/matlabclient01/Peaky_Blinders_2/featurescode/imagergbonly.pth'


BATCH_SIZE = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRANSFORM_IMG =transforms.Normalize((0.64, 0.64, 0.64), (0.1906, 0.188, 0.1906)) 
transform1=transforms.ToTensor()

for each in path2:
    image=plt.imread(each) 
    lbp_all_trainnonmitosis=image
    lbp_train.append(lbp_all_trainnonmitosis)


for each in path3:
    image=plt.imread(each)
    lbp_all_trainmitosis=image
    lbp_train.append(lbp_all_trainmitosis)


for each in path4:
    image=plt.imread(each)
    lbp_all_valnonmitosis=image
    lbp_val.append(lbp_all_valnonmitosis)


for each in path5:
    image=plt.imread(each)
    lbp_all_valmitosis=image
    lbp_val.append(lbp_all_valmitosis)

f1list=[]

import torch.utils.data as utils

  # a list of numpy arrays
labels_nonmitosis=torch.ones(len(path2)) # another list of numpy arrays (targets)
labels_mitosis=torch.zeros(len(path3))
train_labels=torch.cat((labels_nonmitosis,labels_mitosis))
y1_tensor = torch.tensor(train_labels, dtype=torch.long)
tensor_x1 = torch.stack([transform1(i) for i in lbp_train]) # transform to torch tensors
tensor_x4=torch.stack([TRANSFORM_IMG(i) for i in tensor_x1])

train_data = utils.TensorDataset(tensor_x4,y1_tensor) 

train_data_loader = utils.DataLoader(train_data,batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # create your dataloader


labels_nonmitosis_val=torch.ones(len(path4)) # another list of numpy arrays (targets)
labels_mitosis_val=torch.zeros(len(path5))
val_labels=torch.cat((labels_nonmitosis_val,labels_mitosis_val))
y2_tensor = torch.tensor(val_labels, dtype=torch.long)

tensor_x2 = torch.stack([transform1(i) for i in lbp_val]) # transform to torch tensors
tensor_x3=torch.stack([TRANSFORM_IMG(i) for i in tensor_x2])
test_data = utils.TensorDataset(tensor_x3,y2_tensor) # create your datset
test_data_loader = utils.DataLoader(test_data,batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


trainset = train_data
trainloader = train_data_loader 

testset =test_data 
testloader = test_data_loader
total_test_data=len(test_data)
total_train_data=len(train_data)
classes = ('mitosis','nonmitosis')
f11=0




import torch.nn as nn
import torch.nn.functional as F

#padding=k-1/2
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1 ,stride=1)
        self.bn1=nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2=nn.Conv2d(16,32,3,padding=1,stride=1)
        self.bn2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32,32,3,padding=1,stride=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(32, 64, 3,padding=1,stride=1)
        # 0.2
        self.bn3=nn.BatchNorm2d(64)        
        self.conv5 = nn.Conv2d(64,64,3,padding=1,stride=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv6 = nn.Conv2d(64,128,3,padding=1,stride=1)
        # 0.3
        self.bn4=nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128,128,3,padding=1,stride=1)
        self.pool = nn.MaxPool2d(2, 2)   
        self.conv8 = nn.Conv2d(128,256,3,padding=1,stride=1)
        # 0.4
        self.bn5=nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256,256,3,padding=1,stride=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv10 = nn.Conv2d(256,512,3,padding=1,stride=1)
        # 0.5
        self.bn6=nn.BatchNorm2d(512)
        self.pool=nn.MaxPool2d(2,2)
        self.dropout=nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(512 * 40 * 40, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x=F.leaky_relu(self.bn1(x))
        x = F.leaky_relu(self.conv2(x))
        x=F.leaky_relu(self.bn2(x))
        x =F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x=F.leaky_relu(self.bn3(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x=F.leaky_relu(self.bn4(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x=F.leaky_relu(self.bn5(x))
        x = F.leaky_relu(self.conv9(x))
        x = F.leaky_relu(self.conv10(x))
        x=F.leaky_relu(self.bn6(x))
        x=F.leaky_relu(self.dropout(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

import torch.optim as optim
net=net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

correct_train=0
for epoch in range(40):  
    print("epoch HL " , epoch)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels_train = data
# zero the parameter gradients
        inputs = inputs.to(device)
        labels_train = labels_train.to(device)        
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels_train)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if(correct_train==0):
            cyriak2=labels_train
        else:
            cyriak2=torch.cat((cyriak2,labels_train))
        outputs = net(inputs)
        _, predicted_train = torch.max(outputs.data, 1)
        if(correct_train==0):
            cyriak12=predicted_train
     
        if(correct_train>0):
            cyriak12=torch.cat((cyriak12,predicted_train))
        correct_train += (predicted_train == labels_train).sum().item()
    print('Train Accuracy : %d %%' % (
        100 * correct_train / total_train_data))
        

    labels_val=np.array(cyriak2.cpu())
    predicted_val=np.array(cyriak12.cpu())
    f1score=f1_score(labels_val,predicted_val,pos_label=0, average='binary')
        
        
    print("Train f1_score",f1score)
    correct_train=0



    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)            
            if(correct==0):
                cyriak=labels
            else:
                cyriak=torch.cat((cyriak,labels))
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            if(correct==0):
                cyriak1=predicted
         
            if(correct>0):
                cyriak1=torch.cat((cyriak1,predicted))
            correct += (predicted == labels).sum().item()
    print('Accuracy : %d %%' % (
        100 * correct / total_test_data))
    
    
    labels_val=np.array(cyriak.cpu())
    predicted_val=np.array(cyriak1.cpu())
    f1score=f1_score(labels_val,predicted_val,pos_label=0, average='binary')

    
    print("f1_score",f1score)
    print('\n')


    f1list.append(f1score)
    f12=f11
    f11=f1score
    if(f11>=f12):
    
        
        torch.save(net.state_dict(), net_PATH)


print('Finished Training')

