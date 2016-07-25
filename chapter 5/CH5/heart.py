# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:26:02 2015

@author: LegendsUser
"""

import numpy as np
import random
from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import sklearn.svm
from scikitWQDA import WQDA
from SelfLearning import SelfLearningModel

heart = fetch_mldata("heart")
X = heart.data
ytrue = np.copy(heart.target)
ytrue[ytrue==-1]=0

labeled_N = 2
ys = np.array([-1]*len(ytrue)) # -1 denotes unlabeled point
random_labeled_points = random.sample(list(np.where(ytrue == 0)[0]),
                                      int(labeled_N/2))+random.sample(list(np.where(ytrue == 1)[0]), int(labeled_N/2))

ys[random_labeled_points] = ytrue[random_labeled_points]

basemodel = SGDClassifier(loss='log', penalty='l1') 

basemodel.fit(X[random_labeled_points, :], ys[random_labeled_points])
print("supervised log.reg. score", basemodel.score(X, ytrue))

ssmodel = SelfLearningModel(basemodel)
ssmodel.fit(X, ys)
print("self-learning log.reg. score", ssmodel.score(X, ytrue))
