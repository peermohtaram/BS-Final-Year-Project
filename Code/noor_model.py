import matplotlib.pyplot as plt
import cv2
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

from skimage.color import rgb2gray
#astype('float32')

lbp_val=[]
lbp_train=[]
METHOD='uniform'
path2 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/dataset/train/nonmitosis/*.jpg')
path3 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/dataset/train/mitosis/*.jpg')
path4 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/dataset/val/nonmitosis/*.jpg')
path5 = glob.glob('/home/matlabclient01/Peaky_Blinders_2/dataset/val/mitosis/*.jpg')

net_PATH='/home/matlabclient01/Peaky_Blinders_2/featurescode/savedmodelsab/noor_model.pth'
net_PATH2='/home/matlabclient01/Peaky_Blinders_2/featurescode/savedmodelsab/noor_model2.pth'

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

BATCH_SIZE = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRANSFORM_IMG =transforms.Normalize((0.64, 0.64, 0.64,0.092148684, 0.055051066), (0.1906, 0.188, 0.1906,0.2618006,0.19308178)) 
transform1=transforms.ToTensor()
# settings for LBP

for each in path2:
    image=plt.imread(each) 
    lbp_all_trainnonmitosis=makefeature(image)
    lbp_train.append(lbp_all_trainnonmitosis)


for each in path3:
    image=plt.imread(each)
    lbp_all_trainmitosis=makefeature(image)
    lbp_train.append(lbp_all_trainmitosis)


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




import torch.nn as nn
import torch.nn.functional as F

#padding=k-1/2
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

import torch.optim as optim
net=net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.00005)

correct_train=0
f1_previous = 0
for epoch in range(150):  
    print("epoch HL " , epoch+1)

    running_loss = 0.0
    train_f1 = 0
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
    train_f1 = f1score
        
        
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

    if (f1score >= f1_previous):
        f1_previous = f1score
        torch.save(net.state_dict(), net_PATH)
        
    if (f1score >= f1_previous and train_f1 < 0.95):
        torch.save(net.state_dict(), net_PATH2)
        

    
    print("f1_score",f1score)
    print('\n')


    f1list.append(f1score)
print('Finished Training')

