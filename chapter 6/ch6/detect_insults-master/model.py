from scipy.sparse import csr_matrix, vstack
from scipy import sparse
from numpy import array, zeros, ones, matrix, concatenate
import math
import random

from sklearn import linear_model
from sklearn import cross_validation 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from LM import KneserNeyLM, ngrams
from utils import *
from data import Row

#
# Logistic regression model
class LogRegModel:
  def __init__(self, extractor, C = 7, estimator = None):
    self.extractor = extractor
    if None == estimator:
      self.estimator = linear_model.LogisticRegression(C = C)
    else:
      self.estimator = estimator

  def train(self, rows):
    self.extractor.train(rows)

    rowsFeat = [(row.insult, self.extractor.extract(row)) for row in rows]
    self.featDict = featuresDict(rowsFeat)
    X, Y = featuresToMatrix(self.featDict, rowsFeat)
    self.estimator.fit(X, Y)
    return X, Y

  def classify1(self, row):
    x = featuresVec(self.featDict, self.extractor.extract(row))
    w0, w1 = self.estimator.predict_proba(x)[0]
    return (1.0+w1-w0)/2

  def classify(self, rows):
    X = self.featurize(rows)
    Y = self.estimator.predict_proba(X)
    return array([(1.0+w1-w0)/2 for w0, w1 in Y])

  def featurize(self, rows):
    return array([featuresVec(self.featDict, self.extractor.extract(row)) for row in rows])

class RankModel:
  def __init__(self, extractor, C = 1, estimator = None, n = 10000):
    self.extractor = extractor
    if None == estimator:
      self.estimator = linear_model.LogisticRegression(C = C)
    else:
      self.estimator = estimator

    self.n = n

  def train(self, rows):
    self.extractor.train(rows)

    rowsFeat = [(row.insult, self.extractor.extract(row)) for row in rows]
    self.featDict = featuresDict(rowsFeat)
    posX, posY = featuresToMatrix(self.featDict, filter(lambda (insult, f): insult, rowsFeat))
    negX, negY = featuresToMatrix(self.featDict, filter(lambda (insult, f): not insult, rowsFeat))

    # generate pairwise ranking training set
    posi, negi = range(len(posY)), range(len(negY))
    n = min(len(posi), len(negi))
    Xs, Ys = list(), list()
    while len(Xs)*n < self.n:
      # pos > neg
      random.shuffle(posi)
      random.shuffle(negi)
      Xs.append(posX[posi[:n]] - negX[negi[:n]])
      Ys.append(ones(n))

      # neg < pos
      random.shuffle(posi)
      random.shuffle(negi)
      Xs.append(negX[negi[:n]] - posX[posi[:n]])
      Ys.append(zeros(n))

    X = vstack(Xs)
    Y = concatenate(Ys)

    self.estimator.fit(X, Y)

    return X, Y

  def classify1(self, row):
    x = featuresVec(self.featDict, self.extractor.extract(row))
    w0, w1 = self.estimator.predict_proba(x)[0]
    return (1.0+w1-w0)/2

  def classify(self, rows):
    X = self.featurize(rows)
    Y = self.estimator.predict_proba(X)
    return array([(1.0+w1-w0)/2 for w0, w1 in Y])

  def featurize(self, rows):
    return array([featuresVec(self.featDict, self.extractor.extract(row)) for row in rows])

class RegModel:
  def __init__(self, extractor, estimator):
    self.extractor = extractor
    self.estimator = estimator

  def train(self, rows):
    self.extractor.train(rows)

    rowsFeat = [(row.insult, self.extractor.extract(row)) for row in rows]
    self.featDict = featuresDict(rowsFeat)
    X, Y = featuresToMatrix(self.featDict, rowsFeat)

    self.estimator.fit(X, Y)
    return X, Y

  def classify1(self, row):
    x = featuresVec(self.featDict, self.extractor.extract(row))
    return self.estimator.predict(x)[0]

  def classify(self, rows):
    X = self.featurize(rows)
    Y = self.estimator.predict(X)
    return Y

  def featurize(self, rows):
    return array([featuresVec(self.featDict, self.extractor.extract(row)) for row in rows])

#
# Sentence level classifier
class SenModel:
  def __init__(self, model):
    self.model = model

  def train(self, rows):
    posRows = self._splitSen([row for row in rows if row.insult and 1 == len(row.sentences)])
    negRows = self._splitSen([row for row in rows if not row.insult])
    
    newRows = array(posRows + negRows)

    return self.model.train(newRows)

  def classify1(self, row):
    return self.model.classify(self._splitSen([row])).mean()

  def classify(self, rows):
    return array([self.classify1(row) for row in rows])

  def featurize(self, rows):
    raise "not implemented"

  def _toText(self, sentence):
    return ' '.join([tok[0] for tok in sentence.tokens])

  def _splitSen(self, rows):
    return [Row(row.dt, self._toText(sentence), [sentence], row.insult) for row in rows for sentence in row.sentences]

#
# Language model classifier
class LMModel:
  def __init__(self, seqExtractor, n = 2):
    self.words = None
    self.seqExtractor = seqExtractor
    self.lm0 = KneserNeyLM(n)
    self.lm1 = KneserNeyLM(n)

  def train(self, rows):
    self.seqExtractor.train(rows)
    self.lm0.train([self.seqExtractor.extract(row) for row in rows if not row.insult])
    self.lm1.train([self.seqExtractor.extract(row) for row in rows if row.insult])

  def classify1(self, row):
    seq = self.seqExtractor.extract(row)
    w = 0.0 + self.lm1.score(seq) - self.lm0.score(seq)
    if 100 < w:  w = 100
    if w < -100: w = -100
    return 1.0/(1.0 + math.exp(-w))

  def classify(self, rows):
    return array([self.classify1(row) for row in rows])

# LM sequence extractors
class WordsSeqExtractor():
  def __init__(self, seqF):
    self.seqF = seqF
    self.words = dict()

  def train(self, rows):
    self.words = self.prepareWords(rows)

  def extract(self, row):
    def getWord(w):
      if w in self.words:
        return w
      else:
        return '<unk>'
      
    return filter(lambda s: s != None, map(getWord, self.seqF(row)))

  def prepareWords(self, rows):
    words = buildDict(((row.insult, self.seqF(row)) for row in rows))
    return dict([(w, (0.0+pc/(1.0+pc+nc))) for (w, (pc, nc)) in words.items() if 5 < pc+nc and 3 <= len(w)])

# Basic sequence extractors
class SeqExtractor():
  def __init__(self, seqF):
    self.seqF = seqF

  def train(self, rows):
    pass

  def extract(self, row):
    return self.seqF(row)

class Basic():
  def __init__(self, extractor):
    self.extractor = extractor

  def train(self, rows):
    pass

  def classify(self, rows):
    return array([self.extractor(row) for row in rows])

  def classify1(self, row):
    return self.extractor(row)

  def featurize(self, rows):
    return self.classify(rows)

  def featurize1(self, row):
    return self.classify1(row)

#
# Extract features set (like words/ngrams/shapes)
class SetExtractor:
  def __init__(self, extractF, scoreF):
    self.extractF = extractF
    self.scoreF = scoreF

    self.featDict = dict()

  def train(self, rows):
    featRows = ((row.insult, self.extractF(row)) for row in rows)
    self.featDict = self.scoreF(featRows)

  def extract(self, row):
    features = set(self.extractF(row))
    return [(feature, self.featDict[feature]) for feature in features if feature in self.featDict]

#
# Extract features, score with TF (normalized to length)
class TFExtractor:
  def __init__(self, extractF):
    self.extractF = extractF

  def train(self, rows):
    pass

  def extract(self, row):
    fs = self.extractF(row)
    l = len(fs)
    if 0 == l:
      return []
    return [(f, (0.0+c)/l) for f, c in count(fs)]

class CombinedExtractor:
  def __init__(self, *extractors):
    self.extractors = extractors
  
  def train(self, rows):
    for e in self.extractors:
      e.train(rows)

  def extract(self, row):
    return [(f, w) for e in self.extractors for f, w in e.extract(row)]

# decorate feature
def decorateFeat(mask, extractF):
  return lambda row: [mask % feat for feat in extractF(row)]

# Features: combinators
def textFeat(extractF):
  return lambda row: extractF(row.text)

# Combine feature extractor methods
def combine(*fs):
  def exF(row):
    features = list()
    for f in fs:
      features.extend(f(row))
    return features
  return exF

# (res, [features]) -> {feature: i}
def featuresDict(rows):
  featSet = set([f for res, features in rows for f, w in features])
  return dict([(f, i) for i, f in enumerate(featSet)])
  
# FeatDict -> (res, [features]) -> Matrix Double
def featuresToMatrix(featDict, rows):
  n = len(rows)
  X = csr_matrix([featuresVec(featDict, features) for res, features in rows])
  Y = array([float(res) for res, features in rows])
  return X, Y

def featuresVec(featuresDict, features):
  n = len(featuresDict)
  v = zeros(n)
  for f, w in features:
    featI = featuresDict.get(f, None)
    if None != featI:
      v[featI] = w
  return v

# Scores
def probScore(featRows):
  return dict([(feat, (0.0+pc/(1.0+pc+nc))) for (feat, (pc, nc)) in buildDict(featRows).items() if 0 < pc and 5 < pc+nc])

def oneScore(featRows):
  return dict([(feat, 1) for (feat, (pc, nc)) in buildDict(featRows).items() if 0 < pc and 5 < pc+nc])

# calculate feature-class distr
# input: [(class, [feature])]
def buildDict(recs):
  words = dict()
  for outcome, features in recs:
    for feature in features:
      posCount, negCount = words.get(feature, (0, 0))
      if outcome:
        posCount += 1
      else:
        negCount += 1
      words[feature] = (posCount, negCount)

  return words
