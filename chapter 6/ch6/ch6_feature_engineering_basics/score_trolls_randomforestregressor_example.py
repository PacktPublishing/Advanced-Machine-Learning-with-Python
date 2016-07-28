# this script provides a demonstration of how the cleaned, vectorized feature set might be incorporated into a model.
# for a performant model that takes a much bigger & optimised range of input features, consult the model provided in the "detect insults - kaggle competitor version" directory.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc

# this genfromtxt and read_csv involvement is necessary as we exported the script from our cleaning script so that we could examine it. It's clumsy and a non-demonstrative implementation would simply call the cleaner function from ch6_simplemodel_cleaning_v5.py


train_x = np.genfromtxt('train_x.csv', delimiter = ',')
test_x = np.genfromtxt('test_x.csv', delimiter = ',')
training = pd.read_csv('trainingtrolls.csv', header=True, names=['y', 'date', 'Comment', 'usage'])
test = pd.read_csv('testtrolls.csv', header=True, names=['y', 'date', 'Comment'])

# we define our scores using the scores in the training and test data
        

train_y = training["y"]
test_y = test["y"]


# now we'll take those terms and build, then fit, a simplistic random forest implementation

rf = RandomForestRegressor(n_estimators = 1000, max_depth = 10, max_features = 1000)


rf.fit(train_x, train_y)
y_submission = rf.predict(test_x)[:,1]

y_submission = (y_submission - y_submission.min())/(y_submission.max() - y_submission.min())

print("Random Forest benchmark score, 1000 estimators")
print(rf.score(test_x, test_y))

fpr, tpr, _ = roc_curve(y_submission, test_y)
roc_auc = auc(fpr, tpr)
print("Random Forest benchmark AUC, 1000 estimators")
print(roc_auc)
