# this script provides a demonstration of how the cleaned, vectorized feature set might be incorporated into a model.
# for a performant model that takes a much bigger & optimised range of input features, consult the model provided in the "detect insults - kaggle competitor version" directory.
# this model structure is an adaptation of a network architecture originally written by Emanuele Olivetti. For a better view of how this kind of ensemble architecture will perform with the right data, consult the "ch8_competitive_kaggle_ensemble" script in the Chapter 8 code directory.

import pandas as pd
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# this genfromtxt and read_csv involvement is necessary as we exported the script from our cleaning script so that we could examine it. It's clumsy and a non-demonstrative implementation would simply call the cleaner function from ch6_simplemodel_cleaning_v5.py

train_x = np.genfromtxt('train_x.csv', delimiter = ',')
test_x = np.genfromtxt('test_x.csv', delimiter = ',')

training = pd.read_csv('trainingtrolls.csv', header=True, names=['y', 'date', 'Comment', 'usage'])
test = pd.read_csv('testtrolls.csv', header=True, names=['y', 'date', 'Comment'])

# we define our scores using the scores in the training and test data
    
train_y = training["y"]
test_y = test["y"]

# this is our workhorse function; it creates a set of models and blends them into a single set of predictions.

if __name__ == '__main__':

    np.random.seed(0)

    n_folds = 10
    verbose = True

    #we're using k-fold cross validation     
    
    skf = list(StratifiedKFold(train_y, n_folds))
    
    # here we're trying a simple set of random forest classifiers.

    clfs = [RandomForestClassifier(n_estimators=1, n_jobs=-50,  
        criterion='gini'),
            RandomForestClassifier(n_estimators=1, n_jobs=-50,   
            criterion='entropy'),
            ExtraTreesClassifier(n_estimators=1, n_jobs=-50, 
            criterion='gini'),
            ExtraTreesClassifier(n_estimators=1, n_jobs=-50, 
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
            y_submission = clf.predict_proba(X_train)[:,1]
            dataset_blend_train[y_train, j] = y_submission
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

#the following code generates a roc curve with AUC measure.

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

