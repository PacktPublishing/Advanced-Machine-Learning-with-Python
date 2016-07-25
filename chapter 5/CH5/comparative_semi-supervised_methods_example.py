import numpy as np
import random
import matplotlib.pyplot as plt
from frameworks.CPLELearning import CPLELearningModel
from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import sklearn.svm
from methods.scikitWQDA import WQDA
from frameworks.SelfLearning import SelfLearningModel

# load data
heart = fetch_mldata("heart")
X = heart.data
ytrue = np.copy(heart.target)
ytrue[ytrue==-1]=0

# label a few points 
labeled_N = 30
ys = np.array([-1]*len(ytrue)) # -1 denotes unlabeled point
random_labeled_points = random.sample(np.where(ytrue == 0)[0], labeled_N/2)+\
                        random.sample(np.where(ytrue == 1)[0], labeled_N/2)
ys[random_labeled_points] = ytrue[random_labeled_points]

# supervised score 
basemodel = WQDA() # weighted Quadratic Discriminant Analysis
#basemodel = SGDClassifier(loss='log', penalty='l1') # scikit logistic regression
basemodel.fit(X[random_labeled_points, :], ys[random_labeled_points])
#print "supervised log.reg. score", basemodel.score(X, ytrue)

# fast (but naive, unsafe) self learning framework
ssmodel = SelfLearningModel(basemodel)
ssmodel.fit(X, ys)
print("this is the fitted thing", ssmodel.fit(X,ys))
y_score = ssmodel.predict(heart.data)
#print "heart.target", heart.target
#print "this is the prediction", y_score
print("self-learning log.reg. score", ssmodel.score(X, ytrue))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(label_binarize(heart.target, classes = [0,1]), label_binarize(y_score, classes = [0,1]))
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(2):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for self-learning log.reg. classification of the Heart dataset')
plt.legend(loc="lower right")
plt.show()


# semi-supervised score (base model has to be able to take weighted samples)
ssmodel = CPLELearningModel(basemodel)
ssmodel.fit(X, ys)
y_score = ssmodel.predict(heart.data)
print("CPLE semi-supervised log.reg. score", ssmodel.score(X, ytrue)))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(label_binarize(heart.target, classes = [0,1]), y_score)
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(2):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for CPLE semi-supervised log.reg. classification of the Heart dataset')
plt.legend(loc="lower right")
plt.show()


# semi-supervised score, WQDA model
ssmodel = CPLELearningModel(WQDA(), predict_from_probabilities=True) # weighted Quadratic Discriminant Analysis
ssmodel.fit(X, ys)
y_score = ssmodel.predict(heart.data)
print("CPLE semi-supervised WQDA score", ssmodel.score(X, ytrue))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(label_binarize(heart.target, classes = [0,1]), y_score)
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(2):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for CPLE semi-supervised WQDA classification of the Heart dataset')
plt.legend(loc="lower right")
plt.show()

# semi-supervised score, RBF SVM model
ssmodel = CPLELearningModel(sklearn.svm.SVC(kernel="rbf", probability=True), predict_from_probabilities=True) # RBF SVM
y_score = ssmodel.fit(X, ys)
y_score = ssmodel.predict(heart.data)
print("CPLE semi-supervised RBF SVM score", ssmodel.score(X, ytrue)


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(label_binarize(heart.target, classes = [0,1]), y_score)
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(2):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for semi-supervised RBF SVM classification of the Heart dataset')
plt.legend(loc="lower right")
plt.show()


#C:\Users\LegendsUser\Anaconda\Lib;C:\Users\LegendsUser\Anaconda\DLLs