#!/usr/bin/env python

import random
from operator import itemgetter
import math
from datetime import datetime

import numpy

from sklearn import linear_model
from sklearn import cross_validation 
from sklearn import metrics
from sklearn import ensemble

from model import Basic
from mkhtml import *
from utils import *
import data
import os.path

class StackModel:
  def __init__(self, n_estimators = 50, k = 4, n = 2):
    self.classifierFs = [
      # stem
      stemLogistic,
      stem12Logistic,
      stem12SortedLogistic,
      stemRank,
      stem12Rank,
      stem12SortedRank,

      # words
      wordsLogistic,
      words2Logistic,
      words12Logistic,
      words12SortedLogistic,

      # tags 
      tagsLogistic,
      tags12Logistic,
      tags12TfLogistic,

      # word shapes
      wordShapesLogistic,
      wordShapesTfLogistic,

      # subseq
      stemsSubseq2Logistic_5,
      stemsSubseq2Logistic_6,
      stemsSubseq2SortedLogistic_5,
      stemsSubseq2SortedLogistic_6,

      stemsSubseq3Logistic_5,
      stemsSubseq3Logistic_6,
      stemsSubseq3SortedLogistic_5,

      tagsSubseq2Logistic_5,
      tagsSubseq2Logistic_6,
      tagsSubseq2SortedLogistic_5,
      tagsSubseq2SortedLogistic_6,
      tagsSubseq3Logistic_4,
      tagsSubseq3Logistic_5,
      tagsSubseq3Logistic_6,

      # char ngrams
      ngramsLogistic_3,
      ngramsLogistic_4,

      ngramsTfLogistic_3,
      ngramsTfLogistic_4,

      ngramsRank_3,
      ngramsRank_4,

      ngramsTfRank_3,
      ngramsTfRank_4,

      ngramsLogisticSen_2,
      ngramsLogisticSen_3,
      ngramsLogisticSen_4,

      ngramsTfLogisticSen_2,
      ngramsTfLogisticSen_3,
      ngramsTfLogisticSen_4,

      ngramsRankSen_2,
      ngramsRankSen_3,
      ngramsRankSen_4,

      ngramsTfRankSen_2,
      ngramsTfRankSen_3,
      ngramsTfRankSen_4,

      # mixed ngrams
      mixedST,
      mixedTS,
      mixedST_TS,

      mixedSTT,
      mixedTST,
      mixedTTS,
      mixedSTT_TST_TTS,

      # lang models
      stemLm_2,
      stemLm_3,
      stemLm_4,
      stemLm_5,
      stemLm_6,
      stemLm_7,

      stemLmSen_2,
      stemLmSen_3,
      stemLmSen_4,
      stemLmSen_5,
      stemLmSen_6,
      stemLmSen_7,

      tagLm_2,
      tagLm_3,
      tagLm_4,
      tagLm_5,
      tagLm_6,
      tagLm_7,

      tagLmSen_2,
      tagLmSen_3,
      tagLmSen_4,
      tagLmSen_5,
      tagLmSen_6,
      tagLmSen_7,
      
      # syntax
      syntaxStems2,
      syntaxStems12,
      syntaxStems123,
      syntaxStems2Dep,
      syntaxStemsL,
      syntaxStemsR,
      syntaxStemsLR,
      syntaxStems12_LR,

      syntaxTags2,
      syntaxTags3,
      syntaxTags12,
      syntaxTags123,

      syntaxMixedST,
      syntaxMixedTS,
      syntaxMixedST_TS,

      syntaxMixedSTT,
      syntaxMixedTST,
      syntaxMixedTTS,
      syntaxMixedSTT_TST_TTS,

      syntaxStems12Sen,
      syntaxStemsRSen,
      syntaxStemsLRSen,
      syntaxStems12_LRSen,

      # basic features
      logLen,
      logSentencesCount,
      upperPortion,
      lowerPortion,
      digitPortion,
      puncPortion,
      otherPortion,
    ]

    self.classifiers = None

    self.estimator = ensemble.ExtraTreesRegressor(n_estimators = n_estimators, compute_importances = True, n_jobs = -1)

    self.k = k
    self.n = n

  def train(self, rows):
    Xs = []
    Ys = []
    rows = numpy.array(rows)
    for i in range(self.n):
      kfold = cross_validation.KFold(len(rows), k = self.k, indices = True, shuffle = True)
      for k, (train, test) in enumerate(kfold):
        rowsTrain = rows[train]
        rowsTest = rows[test]

        Xk = numpy.column_stack([self._train1(i, k, ci, rowsTrain, rowsTest) for ci in range(len(self.classifierFs))])
        Yk = numpy.array([float(row.insult) for row in rowsTest])
        Xs.append(Xk)
        Ys.append(Yk)
    X = numpy.vstack(Xs)
    Y = numpy.concatenate(Ys)

    dt = datetime.now()
    print 'fit estimator, X=%s Y=%s' % (X.shape, Y.shape)
    self.estimator.fit(X, Y)
    print 'fit estimator, done %s' % str(datetime.now() - dt)

    dt = datetime.now()
    print 'final train'
    self.classifiers = [f() for f in self.classifierFs]
    for f, e in zip(self.classifierFs, self.classifiers):
      dt1 = datetime.now()
      e.train(rows)
      print 'final train, %s %s' % (f.func_name, str(datetime.now() - dt1))

    print 'final train, done %s' % str(datetime.now() - dt)

    print 'features'
    print '\n'.join(['%s\t%f' % (f.func_name, fimp) for f, fimp in zip(self.classifierFs, self.estimator.feature_importances_)])

    return X, Y

  def _train1(self, n, k, i, trainRows, testRows):
    dt = datetime.now()

    f = self.classifierFs[i]
    name = f.func_name
    e = f()

    e.train(trainRows)
    X = e.classify(testRows)

    print 'trained, n=%d k=%d %s %s' % (n, k, name, str(datetime.now() - dt))
    return X

  def classify1(self, row):
    x = self.featurize1(row)
    return self.estimator.predict(x)

  def classify(self, rows):
    X = self.featurize(rows)
    return self.estimator.predict(X)

  def featurize(self, rows):
    X = numpy.column_stack([e.classify(rows) for e in self.classifiers])
    return X

  def featurize1(self, row):
    X = numpy.array([e.classify1(row) for e in self.classifiers])
    return X
