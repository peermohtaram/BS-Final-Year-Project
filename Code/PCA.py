#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: matlabclient01
"""

import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.misc import imread

labels = []
images = pd.DataFrame()
for each in glob.iglob('/home/matlabclient01/Peaky_Blinders_2/dataset/train/mitosis/*.jpg'):
    img = imread(each)
    face = pd.Series(img.flatten())
    images = images.append(face,ignore_index=True)
    labels.append(1)

for each in glob.iglob('/home/matlabclient01/Peaky_Blinders_2/dataset/train/nonmitosis/*.jpg'):
    img = imread(each)
    face = pd.Series(img.flatten(), name = each)
    images = images.append(face,ignore_index=True)    
    labels.append(0)

#np.concatenate((images, labels), axis= 1)
   
pca = PCA(n_components=3)

pca_result = pca.fit_transform(images)
pca_one = pca_result[:,0]
pca_two = pca_result[:,1]
pca_three = pca_result[:,2]

plt.figure(figsize=(10,10))
sns.scatterplot(
    x=pca_one,
    y=pca_two,
    hue = labels,
    palette=sns.color_palette("hls", 2),
    legend="full",
    alpha=0.8
)
