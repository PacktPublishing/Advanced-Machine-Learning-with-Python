# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 08:45:43 2015

@author: a-johear
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.1, 1.5, (2, 10))
cluster2 = np.random.uniform(4.2, 1.9, (2,10))
X = np.hstack((cluster1, cluster2)).T

K = range(1, 10)
meandistortions = []
for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis = 1))/ X.shape[0])
        
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting K w/ Elbow Method')
plt.show()
