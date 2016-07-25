# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 01:59:23 2016

@author: LegendsUser
"""

import pandas as pd
from clean_trolls import cleaner
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


trolls = pd.read_csv('trolls.csv', header=True, names=['y', 'date', 'Comment'])

# UNLEASH THE TRAINING KRAKEN
        
train_x = trolls["Comment"].apply(cleaner)

vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 3946)
train_y = trolls["y"]
train_x = vectorizer.fit_transform(train_x)
train_x = train_x.toarray()


moretrolls = pd.read_csv('moretrolls.csv', header=True, names=['y', 'date', 'Comment', 'Usage'])

# UNLEASH THE KRAKEN

test_x = moretrolls["Comment"].apply(cleaner)
test_y = moretrolls["y"]
test_x = vectorizer.fit_transform(test_x)
test_x = test_x.toarray()

if __name__ == '__main__':

    np.random.seed(0)

    n_folds = 4
    verbose = True
    
    skf = list(StratifiedKFold(train_y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1,  
        criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1,   
            criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, 
            criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, 
            criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, 
            subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((train_x.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((test_x.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((test_x.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = train_x
            y_train = train_y
            X_test = test_x
            y_test = test_y
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[y_test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_test, test_y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min())/(y_submission.max() - y_submission.min())

    print "Saving Results."
    np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')

fpr, tpr, _ = roc_curve(y_test, y_submission)
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

