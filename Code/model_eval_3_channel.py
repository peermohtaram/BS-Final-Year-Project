#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:33:41 2019

@author: matlabclient01
"""
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
lbp_val=[]
from sklearn.metrics import f1_score
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
import cv2
from skimage.feature import local_binary_pattern
from sklearn.metrics import confusion_matrix
from test_perfomance import pr_curve, roc_curve_plot, plot_confusion_matrix
import pywt
import pywt.data

from skimage.color import rgb2gray


net_PATH='//home/matlabclient01/Peaky_Blinders_2/featurescode/savedmodelsab/imagelbppth.pth'


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
        self.dropout=nn.Dropout2d(p=0.5)
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

model = Net()
model.load_state_dict(torch.load(net_PATH))
model=model.eval()


path4 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/test_data/80x80test_images_nonmitotic/*.jpg')
path5 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/test_data/80x80test_images_groundtruth/*.jpg')
BATCH_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#TRANSFORM_IMG =transforms.Normalize((0.64, 0.64, 0.64,0.092148684), (0.1906, 0.188, 0.1906,0.2618006))
TRANSFORM_IMG =transforms.Normalize((0.64, 0.64, 0.64), (0.1906, 0.188, 0.1906)) 
transform1=transforms.ToTensor()
for each in path4:
    image=plt.imread(each)
    image = resize(image, (80,80,3), mode='constant')
    lbp_all_valnonmitosis = image
    lbp_val.append(lbp_all_valnonmitosis)


for each in path5:
    image=plt.imread(each)
    image = resize(image, (80,80,3), mode='constant')
    lbp_all_valmitosis = image
    lbp_val.append(lbp_all_valmitosis)

f1list=[]

import torch.utils.data as utils


labels_nonmitosis_val=torch.ones(len(path4)) # another list of numpy arrays (targets)
labels_mitosis_val=torch.zeros(len(path5))
val_labels=torch.cat((labels_nonmitosis_val,labels_mitosis_val))
y2_tensor = torch.tensor(val_labels, dtype=torch.long)

tensor_x2 = torch.stack([transform1(i) for i in lbp_val]) # transform to torch tensors
tensor_x3=torch.stack([TRANSFORM_IMG(i) for i in tensor_x2])
test_data = utils.TensorDataset(tensor_x3,y2_tensor) # create your datset
test_data_loader = utils.DataLoader(test_data,batch_size=BATCH_SIZE, shuffle=True, num_workers=4)



testset =test_data 
testloader = test_data_loader
total_test_data=len(test_data)
classes = ('mitosis','nonmitosis')

import torch.optim as optim
model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
        outputs = model(images)
        scores, predicted = torch.max(outputs.data, 1)
#        print(predicted)
        if(correct==0):
            cyriak1=predicted
            cyriak2=scores
            cyriak3=outputs
     
        if(correct>0):
            cyriak1=torch.cat((cyriak1,predicted))
            cyriak2=torch.cat((cyriak2,scores))
            cyriak3=torch.cat((cyriak3,outputs))
        correct += (predicted == labels).sum().item()

        
        
labels_val=np.array(cyriak.cpu())
predicted_val=np.array(cyriak1.cpu())
scores=np.array(cyriak2.cpu())
scores_2=cyriak3.data.cpu().numpy()[:,0]
f1score=f1_score(labels_val,predicted_val,pos_label=0, average='binary')
print('Accuracy : %d %%' % (
    100 * correct / total_test_data))
print("f1_score",f1score)
print('\n')

np.save('true',labels_val) #CHANGE NAME HERE FOR EVERY RUN IF WANT TO UPDATE
np.save('pred',predicted_val)	#CHANGE NAME HERE FOR EVERY RUN IF WANT TO UPDATE
np.save('score',scores_2)	#CHANGE NAME HERE FOR EVERY RUN IF WANT TO UPDATE


#BELOW CODE IS FOR PERFORMANCE GRAPHS AND MATRICES WHICH IMPORTS FROM test_performance.py

#pr_curve(labels_val, scores_2,f1 = f1score, title='Precision-Recall, 5 channelsCNN\nRGB,lbp,hog')
#roc_curve_plot(labels_val, scores_2, title='ROC Curve, 5-Channels CNN\nRGB,lbp,hog' )
#
#cnf_matrix = confusion_matrix(labels_val,predicted_val)
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['Mitosis', 'Non-Mitosis'],
#                      title='Confusion Matrix,  5-Channels CNN\nRGB,lbp,hog')
