#!/usr/bin/env python

import sys
import random
import re
from operator import itemgetter
import math
from itertools import islice
from datetime import datetime

from scipy.sparse import csr_matrix
from numpy import array, zeros, matrix

from sklearn import linear_model
from sklearn import cross_validation 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from data import loadTestJson, loadTrainJson
from LM import KneserNeyLM, ngrams
from wordshape import wordShape, capCount
from features import *
from model import *

# Make submission using model
def mkSub(modelF, trainFile, testFile, outFile):
  printNow("make submission")
  trainRows = list(loadTrainJson(trainFile))
  printNow("  load train rows")

  m = modelF()
  m.train(trainRows)
  printNow("  trained")
  
  makeSubmission(m, testFile, outFile)
  return m

def makeSubmission(model, testFile, outFile):
  rows = loadTestJson(testFile)
  printNow("  load test rows")
  with open(outFile, 'w') as f:
    f.write('Insult,Date,Comment\n')
    f.writelines(['%f,%s,%s\n' % (model.classify1(row), row.dt, row.rawText) for row in rows])
    printNow("  submission created")

def printNow(msg):
  print '%s [%s]' % (msg, datetime.strftime(datetime.now(), '%H:%M:%S'))

def printOutliers(outliers, outFile):
  def insultText(insult):
    if insult: return 'Insult'
    return ''

  def norm(s):
    s = s.replace('\n', '<br>')
    return s

  with open(outFile, 'w') as f:
    f.write('<html><table border="1">\n')
    f.writelines('<tr><td><b>%s</b></td><td><b>%.3f</b></td></tr>\n<tr><td colspan="2">%s</td></tr>\n' % (insultText(insult), score, norm(text)) for diff, insult, score, text in outliers)
    f.write('</table></html>')

def getOutliers(modelF, rows, n = 20, insult = True, k = 4):
  kfold = cross_validation.KFold(len(rows), k = k, indices = True, shuffle = True)
  rows = array(rows)

  train, test = next(islice(kfold, 1))
  model = modelF()
  model.train(rows[train])

  def out(row):
    score = model.classify1(row)
    return (abs(float(row.insult) - score), row.insult, score, row.text)
  return sorted(map(out, filter(lambda row: insult == row.insult, rows[test])), key = itemgetter(0), reverse = True)[:n]

def printOutliersRF(outliers, outFile):
  def insultText(insult):
    if insult: return 'Insult'
    return ''

  def norm(s):
    s = s.replace('\n', '<br>')
    return s

  with open(outFile, 'w') as f:
    f.write('<html><table border="1">\n')
    f.writelines('<tr><td><b>%s</b></td><td><b>%.3f</b></td><td>%s</td></tr>\n<tr><td colspan="3">%s</td></tr>\n' % (insultText(insult), score, ' '.join(['<b>%.3f</b>' % f for f in features]), norm(text)) for diff, insult, score, features, text in outliers)
    f.write('</table></html>')

def getOutliersRF(model, rows, n = 20, insult = True):
  def out(row):
    score = model.classify1(row.text)
    return (abs(float(row.insult) - score), row.insult, score, model.featurize(row.text), row.text)
  return sorted(map(out, filter(lambda row: insult == row.insult, rows)), key = itemgetter(0), reverse = True)[:n]

#
# Print important features
def topFeatures(model, n = 10, reverse = 1):
  return sorted([(word, weight) for ((word, c), weight) in zip(model.featDict.items(), model.estimator.coef_[0])], key = itemgetter(1), reverse = reverse)[:n]

# some models
def wordsLogistic():
  return LogRegModel(SetExtractor(decorateFeat('w[%s]', textFeat(getWords)), probScore))

def words2Logistic():
  return LogRegModel(SetExtractor(decorateFeat('n2[%s]', textFeat(getNgrams(2))), probScore))

def words12Logistic():
  f1 = decorateFeat('w[%s]', textFeat(getWords))
  f2 = decorateFeat('n2[%s]', textFeat(getNgrams(2)))
  return LogRegModel(SetExtractor(combine(f1, f2), probScore))

def words12SortedLogistic():
  f1 = decorateFeat('w[%s]', textFeat(getWords))
  f2 = decorateFeat('n2[%s]', textFeat(getNgramsSorted(2)))
  return LogRegModel(SetExtractor(combine(f1, f2), probScore))

def ngramsLogistic(n):
  mask = 'ss' + str(n) + '[%s]'
  return lambda: LogRegModel(SetExtractor(decorateFeat(mask, textFeat(getCharNgrams(n))), probScore))

def ngramsRank(n):
  mask = 'ss' + str(n) + '[%s]'
  return lambda: RankModel(SetExtractor(decorateFeat(mask, textFeat(getCharNgrams(n))), probScore))

def ngramsTfLogistic(n):
  mask = 'ss' + str(n) + '[%s]'
  return lambda: LogRegModel(TFExtractor(decorateFeat(mask, textFeat(getCharNgrams(n)))))

def ngramsTfRank(n):
  mask = 'ss' + str(n) + '[%s]'
  return lambda: RankModel(TFExtractor(decorateFeat(mask, textFeat(getCharNgrams(n)))))

def wordShapesLogistic():
  return LogRegModel(SetExtractor(decorateFeat('shape[%s]', textFeat(wordShapes)), probScore))

def wordShapesTfLogistic():
  return LogRegModel(TFExtractor(decorateFeat('shape[%s]', textFeat(wordShapes))))

def stemLogistic():
  return LogRegModel(SetExtractor(decorateFeat('stem[%s]', getStems1), probScore))

def stemRank():
  return RankModel(SetExtractor(decorateFeat('stem[%s]', getStems1), probScore))

def stem2Logistic():
  f1 = decorateFeat('stem2[%s]', getStemNgrams(2))
  return LogRegModel(SetExtractor(f1, probScore), C = 11)

def stem12Logistic():
  f1 = decorateFeat('stem1[%s]', getStems1)
  f2 = decorateFeat('stem2[%s]', getStemNgrams(2))
  return LogRegModel(SetExtractor(combine(f1, f2), probScore), C = 11)

def stem12Rank():
  f1 = decorateFeat('stem1[%s]', getStems1)
  f2 = decorateFeat('stem2[%s]', getStemNgrams(2))
  return RankModel(SetExtractor(combine(f1, f2), probScore))

def stem12Ridge():
  f1 = decorateFeat('stem1[%s]', getStems1)
  f2 = decorateFeat('stem2[%s]', getStemNgrams(2))
  return RegModel(SetExtractor(combine(f1, f2), probScore), estimator = linear_model.Ridge(alpha = 14))

def stem12SortedLogistic():
  f1 = decorateFeat('stem1[%s]', getStems1)
  f2 = decorateFeat('stem2[%s]', getStemNgramsSorted(2))
  return LogRegModel(SetExtractor(combine(f1, f2), probScore))

def stem12SortedRank():
  f1 = decorateFeat('stem1[%s]', getStems1)
  f2 = decorateFeat('stem2[%s]', getStemNgramsSorted(2))
  return RankModel(SetExtractor(combine(f1, f2), probScore))

def tagsLogistic():
  f1 = decorateFeat('stem1[%s]', getTags)
  return LogRegModel(SetExtractor(f1, probScore))

def tags12Logistic():
  f1 = decorateFeat('tag[%s]', getTags)
  f2 = decorateFeat('tag2[%s]', getTagNgrams(2))
  return LogRegModel(SetExtractor(combine(f1, f2), probScore))

def tags12TfLogistic():
  f1 = decorateFeat('tag[%s]', getTags)
  f2 = decorateFeat('tag2[%s]', getTagNgrams(2))
  return LogRegModel(TFExtractor(combine(f1, f2)))

#
# LM classifiers
def wordsLm(n):
  return lambda: LMModel(WordsSeqExtractor(wordsSeq), n = n)

def stemLm(n):
  return lambda: LMModel(WordsSeqExtractor(getStems), n = n)

def tagLm(n):
  return lambda: LMModel(SeqExtractor(getTags), n = n)

#
# Subseq classifiers
def wordsSubseq2Logistic(n):
  return lambda: LogRegModel(SetExtractor(getSubseq2(wordsSeq, n), probScore))

def wordsSubseq3Logistic(n):
  return lambda: LogRegModel(SetExtractor(getSubseq3(wordsSeq, n), probScore))

def stemsSubseq2Logistic(n):
  return lambda: LogRegModel(SetExtractor(getSubseq2(getStems, n), probScore))

def stemsSubseq2Rank(n):
  return lambda: RankModel(SetExtractor(getSubseq2(getStems, n), probScore))

def stemsSubseq3Logistic(n):
  return lambda: LogRegModel(SetExtractor(getSubseq3(getStems, n), probScore))

def tagsSubseq2Logistic(n):
  return lambda: LogRegModel(SetExtractor(getSubseq2(getTags, n), probScore))

def tagsSubseq3Logistic(n):
  return lambda: LogRegModel(SetExtractor(getSubseq3(getTags, n), probScore))

def stemsSubseq2SortedLogistic(n):
  return lambda: LogRegModel(SetExtractor(getSubseq2Sorted(getStems, n), probScore))

def stemsSubseq2SortedRank(n):
  return lambda: RankModel(SetExtractor(getSubseq2Sorted(getStems, n), probScore))

def stemsSubseq2SortedRank(n):
  return lambda: RankModel(SetExtractor(getSubseq2Sorted(getStems, n), probScore))

def stemsSubseq3SortedLogistic(n):
  return lambda: LogRegModel(SetExtractor(getSubseq3Sorted(getStems, n), probScore))

def tagsSubseq2SortedLogistic(n):
  return lambda: LogRegModel(SetExtractor(getSubseq2Sorted(getTags, n), probScore))

def tagsSubseq3SortedLogistic(n):
  return lambda: LogRegModel(SetExtractor(getSubseq3Sorted(getTags, n), probScore))

#
# Mixed ngrams classifiers
def mixedST():
  return LogRegModel(SetExtractor(mixedNgramsST, probScore))

def mixedTS():
  return LogRegModel(SetExtractor(mixedNgramsTS, probScore))

def mixedST_TS():
  f1 = mixedNgramsST
  f2 = mixedNgramsTS
  return LogRegModel(SetExtractor(combine(f1, f2), probScore))

def mixedSST():
  return LogRegModel(SetExtractor(mixedNgramsSST, probScore))

def mixedSTS():
  return LogRegModel(SetExtractor(mixedNgramsSTS, probScore))

def mixedSTT():
  return LogRegModel(SetExtractor(mixedNgramsSTT, probScore))

def mixedTSS():
  return LogRegModel(SetExtractor(mixedNgramsTSS, probScore))

def mixedTST():
  return LogRegModel(SetExtractor(mixedNgramsTST, probScore))

def mixedTTS():
  return LogRegModel(SetExtractor(mixedNgramsTTS, probScore))

def mixedSTT_TST_TTS():
  f1 = mixedNgramsSTT
  f2 = mixedNgramsTST
  f3 = mixedNgramsTTS
  return LogRegModel(SetExtractor(combine(f1, f2, f3), probScore))

# Syntax classifiers
def syntaxStems12():
  f1 = decorateFeat('stem1[%s]', getStems1)
  f2 = decorateFeat('syn[%s]', synStems2)
  return LogRegModel(SetExtractor(combine(f1, f2), probScore))

def syntaxStems2():
  return LogRegModel(SetExtractor(synStems2, probScore))

def syntaxStems2Dep():
  return LogRegModel(SetExtractor(synStems2Dep, probScore))

def syntaxStemsL():
  return LogRegModel(SetExtractor(synStemsL, probScore))

def syntaxStemsR():
  return LogRegModel(SetExtractor(synStemsR, probScore))

def syntaxStemsLR():
  f1 = decorateFeat('synL[%s]', synStemsL)
  f2 = decorateFeat('synR[%s]', synStemsR)
  return LogRegModel(SetExtractor(combine(f1, f2), probScore))

def syntaxStems12_LR():
  f1 = decorateFeat('stem1[%s]', getStems1)
  f2 = decorateFeat('syn[%s]', synStems2)
  f3 = decorateFeat('synL[%s]', synStemsL)
  f4 = decorateFeat('synR[%s]', synStemsR)
  return LogRegModel(SetExtractor(combine(f1, f2, f3, f4), probScore))

# syntax - from dependency graph
def syntaxStems3():
  return LogRegModel(SetExtractor(synStems3, probScore))

def syntaxStems123():
  f1 = decorateFeat('stem1[%s]', getStems1)
  f2 = decorateFeat('syn2[%s]', synStems2)
  f3 = decorateFeat('syn3[%s]', synStems3)
  return LogRegModel(SetExtractor(combine(f1, f2, f3), probScore))

def syntaxTags2():
  return LogRegModel(SetExtractor(synTags2, probScore))

def syntaxTags3():
  return LogRegModel(SetExtractor(synTags3, probScore))

def syntaxTags12():
  f1 = decorateFeat('tags1[%s]', getTags)
  f2 = decorateFeat('tags2[%s]', synTags2)
  return LogRegModel(SetExtractor(combine(f1, f2), probScore))

def syntaxTags123():
  f1 = decorateFeat('tags1[%s]', getTags)
  f2 = decorateFeat('tags2[%s]', synTags2)
  f3 = decorateFeat('tags3[%s]', synTags3)
  return LogRegModel(SetExtractor(combine(f1, f2, f3), probScore))

def syntaxMixedST():
  return LogRegModel(SetExtractor(synMixedST, probScore))

def syntaxMixedTS():
  return LogRegModel(SetExtractor(synMixedTS, probScore))

def syntaxMixedST_TS():
  f1 = synMixedST
  f2 = synMixedTS
  return LogRegModel(SetExtractor(combine(f1, f2), probScore))

def syntaxMixedSST():
  return LogRegModel(SetExtractor(synMixedSST, probScore))

def syntaxMixedSTS():
  return LogRegModel(SetExtractor(synMixedSTS, probScore))

def syntaxMixedSTT():
  return LogRegModel(SetExtractor(synMixedSTT, probScore))

def syntaxMixedTSS():
  return LogRegModel(SetExtractor(synMixedTSS, probScore))

def syntaxMixedTST():
  return LogRegModel(SetExtractor(synMixedTST, probScore))

def syntaxMixedTTS():
  return LogRegModel(SetExtractor(synMixedTTS, probScore))

def syntaxMixedSTT_TST_TTS():
  f1 = synMixedSTT
  f2 = synMixedTST
  f3 = synMixedTTS
  return LogRegModel(SetExtractor(combine(f1, f2, f3), probScore))

# syntax - sentence level
def syntaxStems12Sen():
  return SenModel(syntaxStems12())

def syntaxStems2Sen():
  return SenModel(syntaxStems2())

def syntaxStems2DepSen():
  return SenModel(syntaxStems2Dep())

def syntaxStemsLSen():
  return SenModel(syntaxStemsL())

def syntaxStemsRSen():
  return SenModel(syntaxStemsR())

def syntaxStemsLRSen():
  return SenModel(syntaxStemsLR())

def syntaxStems12_LRSen():
  return SenModel(syntaxStems12_LR())

#
# Basic exctractors
def logLen():
  return Basic(lambda row: math.log(1.0 + sum(map(lambda sen: len(sen.tokens), row.sentences))))

def logSentencesCount():
  return Basic(lambda row: math.log(1.0 + len(row.sentences)))

def upperPortion():
  def f(row):
    u, l, d, p, o = capCount(row.text)
    return (1.0 + u)/(1.0 + u + l + d + p + o)
  return Basic(f)

def lowerPortion():
  def f(row):
    u, l, d, p, o = capCount(row.text)
    return (1.0 + l)/(1.0 + u + l + d + p + o)
  return Basic(f)

def digitPortion():
  def f(row):
    u, l, d, p, o = capCount(row.text)
    return (1.0 + d)/(1.0 + u + l + d + p + o)
  return Basic(f)

def puncPortion():
  def f(row):
    u, l, d, p, o = capCount(row.text)
    return (1.0 + p)/(1.0 + u + l + d + p + o)
  return Basic(f)

def otherPortion():
  def f(row):
    u, l, d, p, o = capCount(row.text)
    return (1.0 + o)/(1.0 + u + l + d + p + o)
  return Basic(f)

#
# Combined classifiers

def sub5():
  f1 = getStems1
  f2 = getStemNgrams(2)
  f3 = textFeat(getCharNgrams(2))
  return LogRegModel(SetExtractor(combine(f1, f2, f3), probScore))

def sub8():
  f1 = getStems1
  f2 = getStemNgrams(2)
  e1 = SetExtractor(combine(f1, f2), probScore)

  f3 = textFeat(getCharNgrams(2))
  e2 = TFExtractor(f3)

  return LogRegModel(CombinedExtractor(e1, e2), C = 6)

def sub8Ridge():
  f1 = getStems1
  f2 = getStemNgrams(2)
  e1 = SetExtractor(combine(f1, f2), probScore)

  f3 = textFeat(getCharNgrams(2))
  e2 = TFExtractor(f3)

  return RegModel(CombinedExtractor(e1, e2), estimator = linear_model.Ridge(alpha = 1))

def sub5_1():
  f1 = getStems1
  f2 = getStemNgramsSorted(2)
  f3 = textFeat(getCharNgrams(2))
  return LogRegModel(SetExtractor(combine(f1, f2, f3), probScore))

# specialized
def ngramsLogistic_2():
  return ngramsLogistic(2)()

def ngramsLogistic_3():
  return ngramsLogistic(3)()

def ngramsLogistic_4():
  return ngramsLogistic(4)()

def ngramsTfLogistic_2():
  return ngramsLogistic(2)()

def ngramsTfLogistic_3():
  return ngramsLogistic(3)()

def ngramsTfLogistic_4():
  return ngramsLogistic(4)()

def ngramsRank_2():
  return ngramsLogistic(2)()

def ngramsRank_3():
  return ngramsLogistic(3)()

def ngramsRank_4():
  return ngramsLogistic(4)()

def ngramsTfRank_2():
  return ngramsLogistic(2)()

def ngramsTfRank_3():
  return ngramsLogistic(3)()

def ngramsTfRank_4():
  return ngramsLogistic(4)()

def ngramsLogisticSen_2():
  return SenModel(ngramsLogistic(2)())

def ngramsLogisticSen_3():
  return SenModel(ngramsLogistic(3)())

def ngramsLogisticSen_4():
  return SenModel(ngramsLogistic(4)())

def ngramsTfLogisticSen_2():
  return SenModel(ngramsLogistic(2)())

def ngramsTfLogisticSen_3():
  return SenModel(ngramsLogistic(3)())

def ngramsTfLogisticSen_4():
  return SenModel(ngramsLogistic(4)())

def ngramsRankSen_2():
  return SenModel(ngramsLogistic(2)())

def ngramsRankSen_3():
  return SenModel(ngramsLogistic(3)())

def ngramsRankSen_4():
  return SenModel(ngramsLogistic(4)())

def ngramsTfRankSen_2():
  return SenModel(ngramsLogistic(2)())

def ngramsTfRankSen_3():
  return SenModel(ngramsLogistic(3)())

def ngramsTfRankSen_4():
  return SenModel(ngramsLogistic(4)())

#
# LM
def wordsLm_2():
  return wordsLm(2)()

def wordsLm_3():
  return wordsLm(3)()

def wordsLm_4():
  return wordsLm(4)()

def stemLm_2():
  return stemLm(2)()

def stemLm_3():
  return stemLm(3)()

def stemLm_4():
  return stemLm(4)()

def stemLm_5():
  return stemLm(5)()

def stemLm_6():
  return stemLm(6)()

def stemLm_7():
  return stemLm(7)()

def stemLmSen_2():
  return SenModel(stemLm(2)())

def stemLmSen_3():
  return SenModel(stemLm(3)())

def stemLmSen_4():
  return SenModel(stemLm(4)())

def stemLmSen_5():
  return SenModel(stemLm(5)())

def stemLmSen_6():
  return SenModel(stemLm(6)())

def stemLmSen_7():
  return SenModel(stemLm(7)())

def tagLm_2():
  return tagLm(2)()

def tagLm_3():
  return tagLm(3)()

def tagLm_4():
  return tagLm(4)()

def tagLm_5():
  return tagLm(5)()

def tagLm_6():
  return tagLm(6)()

def tagLm_7():
  return tagLm(7)()

def tagLmSen_2():
  return SenModel(tagLm(2)())

def tagLmSen_3():
  return SenModel(tagLm(3)())

def tagLmSen_4():
  return SenModel(tagLm(4)())

def tagLmSen_5():
  return SenModel(tagLm(5)())

def tagLmSen_6():
  return SenModel(tagLm(6)())

def tagLmSen_7():
  return SenModel(tagLm(7)())

# subseq2
def stemsSubseq2Logistic_5():
  return stemsSubseq2Logistic(5)()

def stemsSubseq2Logistic_6():
  return stemsSubseq2Logistic(6)()

def stemsSubseq2SortedLogistic_5():
  return stemsSubseq2SortedLogistic(5)()

def stemsSubseq2SortedLogistic_6():
  return stemsSubseq2SortedLogistic(6)()

def stemsSubseq2Rank_5():
  return stemsSubseq2Rank(5)()

def stemsSubseq2SortedRank_5():
  return stemsSubseq2SortedRank(5)()

def tagsSubseq2Logistic_5():
  return tagsSubseq2Logistic(5)()

def tagsSubseq2Logistic_6():
  return tagsSubseq2Logistic(6)()

def tagsSubseq2SortedLogistic_5():
  return tagsSubseq2SortedLogistic(5)()

def tagsSubseq2SortedLogistic_6():
  return tagsSubseq2SortedLogistic(6)()

# subseq3
def stemsSubseq3Logistic_5():
  return stemsSubseq3Logistic(5)()

def stemsSubseq3Logistic_6():
  return stemsSubseq3Logistic(6)()

def stemsSubseq3SortedLogistic_5():
  return stemsSubseq3SortedLogistic(5)()

def stemsSubseq3SortedLogistic_6():
  return stemsSubseq3SortedLogistic(6)()

def tagsSubseq3Logistic_4():
  return tagsSubseq3Logistic(4)()

def tagsSubseq3Logistic_5():
  return tagsSubseq3Logistic(5)()

def tagsSubseq3Logistic_6():
  return tagsSubseq3Logistic(6)()

def tagsSubseq3SortedLogistic_5():
  return tagsSubseq3SortedLogistic(5)()

def tagsSubseq3SortedLogistic_6():
  return tagsSubseq3SortedLogistic(6)()
