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


net_PATH='//home/matlabclient01/Peaky_Blinders_2/featurescode/savedmodelsab/noor_model.pth'


class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        #### actual Image, intensity, shape, mitosis
        self.conv_layer1=nn.Conv2d(5, 96, 11, stride=4)#, padding=3//2)
        self.relu_layer1=nn.ReLU(True)
        self.pooling_layer1=nn.MaxPool2d(3, stride=2)
        self.batchnorm_layer1=nn.BatchNorm2d(96)
            
        self.conv_layer2=nn.Conv2d(96, 256, 5, stride=1,padding=2)
        self.relu_layer2=nn.ReLU(True)
        self.batchnorm_layer2=nn.BatchNorm2d(256)
            
        self.conv_layer3=nn.Conv2d(256, 512, 3, stride=2,padding=3)
        self.relu_layer3=nn.ReLU(True)
        self.pooling_layer3=nn.MaxPool2d(2, stride=2)
        self.dropout_layer3=nn.Dropout(0.5) # noorulwahab dropout=0.5 
        
        self.conv_layer4=nn.Conv2d(512, 1024, 4, stride=2,padding=0)
        self.relu_layer4=nn.ReLU(True)
        self.dropout_layer4 = nn.Dropout(0.5) # noorulwahab dropout=0.5 
        
        self.conv_layer5=nn.Conv2d(1024, 1024, 3, stride=1)
        self.relu_layer5=nn.ReLU(True)
        self.dropout_layer5 = nn.Dropout(0.5) # noorulwahab dropout=0.5
        self.fc1 = nn.Linear(1024,2)  
        #self.fc2 = nn.Linear(1024,2)  
        
    def forward(self, x):
        x=self.conv_layer1(x)
        x=self.relu_layer1(x)
        x=self.pooling_layer1(x)
        x=self.batchnorm_layer1(x)
        x=self.conv_layer2(x)
        x=self.relu_layer2(x)
        x=self.batchnorm_layer2(x)
        x=self.conv_layer3(x)
        x=self.relu_layer3(x)
        x=self.pooling_layer3(x)
        x=self.dropout_layer3(x)
        x=self.conv_layer4(x)
        x=self.relu_layer4(x)
        x=self.dropout_layer4(x)
        x=self.conv_layer5(x)
        x=self.relu_layer5(x)
        x=self.dropout_layer5(x)
        out=x.view(x.size(0),-1)
        out=self.fc1(out)
        return out

net = Net()

model = Net()
model.load_state_dict(torch.load(net_PATH))
model=model.eval()


path4 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/test_data/80x80test_images_nonmitotic/*.jpg')
path5 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/test_data/80x80test_images_groundtruth/*.jpg')
BATCH_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def makefeature(image):
    METHOD='uniform' 
    radius =18
    n_points = 20
    
    image_up = resize(image, (227,227,3), mode='constant')
    
#   #LBP FEATURES
#    gray_image = rgb2gray(image) #IF BELOW LINE DOESN'T WORK THEN UNCOMMENT THIS
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, METHOD)
   
    #HOG FEATURES
    image1=gray_image
    fd, hog_image = hog(image1, orientations=8, pixels_per_cell=(3, 3),cells_per_block=(3, 3),block_norm='L2-Hys', visualize=True) 
    xyz=np.zeros(144)
    fd=np.append(fd,xyz)
    abc=np.sqrt(len(fd))
    abc=abc.astype('int')
    fd1=fd.reshape(abc,abc)
    fd2=resize(fd1,(227,227),mode='constant')
    
    #BOTH LBP AND HOG FEATURES
#    lbp=lbp.reshape((227,227,1))
    lbp = resize(lbp, (227,227,1), mode='constant')
    fd2=fd2.reshape((227,227,1))

    newimage=np.concatenate((image_up,lbp,fd2),2)

    newimage=newimage.astype('float32')
    return newimage

BATCH_SIZE = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#TRANSFORM_IMG =transforms.Normalize((0.64, 0.64, 0.64,0.092148684), (0.1906, 0.188, 0.1906,0.2618006))
TRANSFORM_IMG =transforms.Normalize((0.64, 0.64, 0.64,0.092148684,0.055), (0.1906, 0.188, 0.1906,0.2618006,0.193)) 
transform1=transforms.ToTensor()
for each in path4:
    image=plt.imread(each)
    lbp_all_valnonmitosis=makefeature(image)
    lbp_val.append(lbp_all_valnonmitosis)


for each in path5:
    image=plt.imread(each)
    lbp_all_valmitosis=makefeature(image)
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

np.save('noor_true',labels_val)		#CHANGE NAME HERE FOR EVERY RUN IF WANT TO UPDATE
np.save('noor_pred',predicted_val)	#CHANGE NAME HERE FOR EVERY RUN IF WANT TO UPDATE
np.save('noor_score',scores_2)		#CHANGE NAME HERE FOR EVERY RUN IF WANT TO UPDATE


#BELOW CODE IS FOR PERFORMANCE GRAPHS AND MATRICES WHICH IMPORTS FROM test_performance.py

#pr_curve(labels_val, scores_2,f1 = f1score, title='Precision-Recall, 5 channelsCNN\nRGB,lbp,hog')
#roc_curve_plot(labels_val, scores_2, title='ROC Curve, 5-Channels CNN\nRGB,lbp,hog' )
#
#cnf_matrix = confusion_matrix(labels_val,predicted_val)
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['Mitosis', 'Non-Mitosis'],
#                      title='Confusion Matrix,  5-Channels CNN\nRGB,lbp,hog')


