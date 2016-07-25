# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:12:00 2015

@author: a-johear
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 12 07:14:04 2015

@author: a-johear
"""

import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.cm as cm

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

#principal component analysis

pca = PCA(n_components=10)
data_r = pca.fit(data).transform(data)

#linear discriminant analysis

lda = LDA(n_components=2)
data_r2 = lda.fit(data, labels).transform(data)

print('explained variance ratio (first ten components): %s' % str(pca.explained_variance_ratio_))
print('sum of explained variance (first ten components): %s' % str(sum(pca.explained_variance_ratio_)))

x = np.arange(2)
ys = [i+x+(i*x)**2 for i in range(10)]

plt.figure()
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
for c, i, target_name in zip(colors, [0,2,3,4,5,6,7,8,9], labels):
    plt.scatter(data_r[labels == i, 0], data_r[labels == i, 1], c=c, alpha = 0.4)
    plt.legend()
    plt.title('Scatterplot of Points plotted in first \n'
    '2 Principal Components')
