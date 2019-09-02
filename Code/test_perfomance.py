#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:26:22 2019

@author: matlabclient01
"""



from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, 5-Channels Boosted CNN')
#
#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('True label') # changed by anabia
    plt.ylabel('Predicted label')  #changed by anabia
    plt.tight_layout()

def pr_curve(y_true, score,label, title='Precision-Recall Curve'):

    precision, recall, _ = precision_recall_curve(y_true, score, pos_label=0)
#    ap = average_precision_score(y_true, score, pos_label=0)
    
    plt.step(recall, precision,label=title, where='post')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)# + '\nF1-measure = %0.2f' % (f1))
#    plt.show()
    
def roc_curve_plot(y_true, scores, title='ROC Curve '):
    #
    fpr, tpr, thresholds=roc_curve(y_true, scores, pos_label=0)
    curve_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=1)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + '\nROC-AUC = %0.2f' % (curve_auc))                                    
#    plt.legend(loc="lower right")
#    plt.show()
#plt.show()