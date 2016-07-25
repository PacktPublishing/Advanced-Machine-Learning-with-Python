# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:02:30 2015

@author: a-johear
"""

import numpy as np
from sklearn.datasets import load_digits


digits = load_digits()
data = digits.data

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target


from som import Som

som = Som(16,16,64,sigma=1.3,learning_rate=0.5)
som.random_weights_init(data)
print("Initialising SOM.")
som.train_random(data,10000) 
print("\n SOM Processing Complete.")

from pylab import plot,axis,show,pcolor,colorbar,bone
bone()
pcolor(som.distance_map().T) 
colorbar()

labels[labels == '0'] = 0
labels[labels == '1'] = 1
labels[labels == '2'] = 2
labels[labels == '3'] = 3
labels[labels == '4'] = 4
labels[labels == '5'] = 5
labels[labels == '6'] = 6
labels[labels == '7'] = 7
labels[labels == '8'] = 8
labels[labels == '9'] = 9

markers = ['o', 'v', '1', '3', '8', 's', 'p', 'x', 'D', '*']
colors = ["r", "g", "b", "y", "c", (0,0.1,0.8), (1,0.5,0), (1,1,0.3), "m", (0.4,0.6,0)]
for cnt,xx in enumerate(data):
 w = som.winner(xx) 
 plot(w[0]+.5,w[1]+.5,markers[labels[cnt]],markerfacecolor='None',
   markeredgecolor=colors[labels[cnt]],markersize=12,markeredgewidth=2)
axis([0,som.weights.shape[0],0,som.weights.shape[1]])
show()



#reduced_data = Som(16,16,64,sigma=1.3,learning_rate=0.5).fit_transform(data)


#from sklearn.cluster import KMeans

#kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
#kmeans.fit(reduced_data)