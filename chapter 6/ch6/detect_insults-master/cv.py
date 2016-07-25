#!/usr/bin/env python

from numpy import array

from sklearn import cross_validation 
from sklearn import metrics

# Crossvalidate using training rows
def crossvalidate(modelF, rows, k = 4, n = 1):
  scores = []
  for i in range(n):
    kfold = cross_validation.KFold(len(rows), k = k, indices = True, shuffle = True)
    rows = array(rows)
    scores.extend([crossScore(modelF, rows[train], rows[test]) for train, test in kfold])
  return ' '.join('%.3f' % score for score in sorted(scores))

def crossScore(modelF, rowsTrain, rowsTest):
  model = modelF()
  model.train(rowsTrain)
  predictions = model.classify(rowsTest)
  fpr, tpr, thresholds = metrics.roc_curve([row.insult for row in rowsTest], [predictions])
  return metrics.auc(fpr, tpr)

def cvPrepare(modelF, rows, k = 4, n = 1):
  for i in range(n):
    kfold = cross_validation.KFold(len(rows), k = k, indices = True, shuffle = True)
    rows = array(rows)
    for train, test in kfold:
      model = modelF()
      Xtrain, Ytrain = model.train(rows[train])
      Xtest = model.featurize(rows[test])
      Ytest = array([float(row.insult) for row in rows[test]])
      yield (model, Xtrain, Ytrain, Xtest, Ytest)

def cvEstimate(modelF, cv):
  scores = []
  for m_, Xtrain, Ytrain, Xtest, Ytest in cv:
    model = modelF()
    model.fit(Xtrain, Ytrain)
    y = model.predict(Xtest)
    fpr, tpr, thresholds = metrics.roc_curve(Ytest, y)
    scores.append(metrics.auc(fpr, tpr))
  scores = array(scores)
  return '%.3f %.3f: %s' % (scores.mean(), scores.std(), ' '.join(['%.3f' % score for score in sorted(scores)]))
