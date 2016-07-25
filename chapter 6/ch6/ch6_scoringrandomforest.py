# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 01:59:23 2016

@author: LegendsUser
"""

import pandas as pd
from clean_trolls import cleaner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#import numpy as np



trolls = pd.read_csv('trolls.csv', header=True, names=['y', 'date', 'Comment'])

# UNLEASH THE TRAINING KRAKEN
        
trolls["Words"] = trolls["Comment"].apply(cleaner)

#trolls.to_csv('trollsout.csv', header=True, names=['y', 'date', 'Comment','Cleaned'])   
    
    
# next we're going to do something a little bit different
# the vectorizer we'll use next uses "term frequency inverse document frequency" vectorization. It turns our text strings into sweet vectors
    
vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 3946)




# now we'll take those terms and build, then fit, a random forest implementation

train_data_features = vectorizer.fit_transform(trolls["Words"])
train_data_features = train_data_features.toarray()
#train_data_features = np.log(train_data_features)

from sklearn.ensemble import RandomForestRegressor

trollspotter = RandomForestRegressor(n_estimators = 1000, max_depth = 10, max_features = 1000)

y = trolls["y"]

trollspotted = trollspotter.fit(train_data_features, y)




# test data

moretrolls = pd.read_csv('moretrolls.csv', header=True, names=['y', 'date', 'Comment', 'Usage'])

# UNLEASH THE KRAKEN

moretrolls["Words"] = moretrolls["Comment"].apply(cleaner)

y = moretrolls["y"]

test_data_features = vectorizer.fit_transform(moretrolls["Words"])
test_data_features = test_data_features.toarray()
#test_data_features = np.log(test_data_features)

# some scoring

#trollspots = trollspotted.predict(test_data_features)

#trollscore = trollspotted.score(test_data_features, y)

trollpredictions = trollspotted.predict(test_data_features)
trollpredictions = (trollpredictions - trollpredictions.min())/(trollpredictions.max() - trollpredictions.min())

fpr, tpr, _ = roc_curve(y, trollpredictions)
roc_auc = auc(fpr, tpr)
print("Random Forest benchmark AUC, 1000 estimators")
print(roc_auc)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
