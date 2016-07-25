from itertools import izip
from operator import itemgetter
from sklearn import metrics, ensemble
import numpy
import multiprocessing

import copy_reg
import types

import data,stack

def reduce_method(m):
  return (getattr, (m.__self__, m.__func__.__name__))

copy_reg.pickle(types.MethodType, reduce_method)


def auc(subFile, testFile):
  ys = []
  with open(subFile) as f1:
    with open(testFile) as f2:
      f = izip(f1, f2)
      next(f) # header

      ys.extend([parseRes(l1, l2) for l1, l2 in f])

  Yact = numpy.array(map(itemgetter(1), ys))
  Yexp = numpy.array(map(itemgetter(0), ys))
  
  fpr, tpr, thresholds = metrics.roc_curve(Yexp, Yact)
  return metrics.auc(fpr, tpr)

def parseRes(lineSub, lineTest):
  subRes = float(lineSub.split(',', 1)[0])
  testRes = float(lineTest.split(',', 1)[0])
  return (testRes, subRes)


class StackEval:
  def __init__(self, rootDir, k = 4, n = 2):
    self.rootDir = rootDir
    trainPath = os.path.join(rootDir, 'train')
    self.m = CachedStackModel(rootDir = trainPath, k = 4, n = 2)

  def prepare(self, trainRows, testRows, trainIndicies = None):
    if None == trainIndicies:
      trainIndicies = array(range(len(trainRows)))

    self.m.train((trainRows, trainIndicies))

    for f, e in zip(self.m.classifierFs, self.m.classifiers):
      self._prepare1(testRows, f, e)

  def _prepare1(self, testRows, f, e):
    tm = datetime.now()

    filePath = os.path.join(self.rootDir, 'test', '%s.npy' % (f.func_name))
    ensureDir(filePath)
    if not os.path.exists(filePath):
      X = e.classify(testRows)
      numpy.save(filePath, X)
      print 'not found [%s], trained %s' % (filePath, str(datetime.now() - tm))

    Y = numpy.array([float(row.insult) for row in testRows])
    numpy.save(os.path.join(self.rootDir, 'test', 'y.npy'), Y)

  def eval(self, estimatorF):
    print 'build train set'

    # build train set
    Xs = []
    Ys = []
    for k, n in [(k, n) for k in range(self.m.k) for n in range(self.m.n)]:
      Xk = numpy.column_stack([self._cachedTrainX(f.func_name, k, n) for f in self.m.classifierFs])
      Yk = numpy.array(self._cachedTrainY(k, n))

      Xs.append(Xk)
      Ys.append(Yk)

    X = numpy.vstack(Xs)
    Y = numpy.concatenate(Ys)

    print 'fit, X.shape=%s Y.shape=%s' % (X.shape, Y.shape)

    e = estimatorF()
    e.fit(X, Y)

    print 'classfy'

    # classify test, calculate AUC
    X = numpy.column_stack([self._cachedTestX(f.func_name) for f in self.m.classifierFs])
    Yexp = self._cachedTestY()
    Yact = e.predict(X)

    fpr, tpr, thresholds = metrics.roc_curve(Yexp, Yact)
    return (metrics.auc(fpr, tpr), e)

  def outliers(self, estimatorF, testRows, n = 30):
    auc, e, X, Yexp, Yact = self.eval(estimatorF)
    recs = zip(testRows, Yexp, Yact)

    fp = sorted(recs, key = lambda rec: rec[1] - rec[2], reverse = False)[:n]
    fn = sorted(recs, key = lambda rec: rec[1] - rec[2], reverse = True)[:n]

    return fp, fn

  def _cachedTrainX(self, name, k, n):
    fileName = os.path.join(self.rootDir, 'train', '%s_k%02d_n%02d.npy' % (name, k, n))
    return numpy.load(fileName)

  def _cachedTrainY(self, k, n):
    fileName = os.path.join(self.rootDir, 'train', 'y_k%02d_n%02d.npy' % (k, n))
    return numpy.load(fileName)

  def _cachedTestX(self, name):
    fileName = os.path.join(self.rootDir, 'test', '%s.npy' % (name))
    return numpy.load(fileName)

  def _cachedTestY(self):
    fileName = os.path.join(self.rootDir, 'test', 'y.npy')
    return numpy.load(fileName)

class CachedStackModel:
  def __init__(self, rootDir, n_estimators = 1000, k = 4, n = 4):
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
      # ngramsLogistic_2,
      ngramsLogistic_3,
      ngramsLogistic_4,

      # ngramsTfLogistic_2,
      ngramsTfLogistic_3,
      ngramsTfLogistic_4,

      # ngramsRank_2,
      ngramsRank_3,
      ngramsRank_4,

      # ngramsTfRank_2,
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

      # mixedSST,
      # mixedSTS,
      mixedSTT,
      # mixedTSS,
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
      # syntaxStems3,
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

      # syntaxMixedSST,
      # syntaxMixedSTS,
      syntaxMixedSTT,
      # syntaxMixedTSS,
      syntaxMixedTST,
      syntaxMixedTTS,
      syntaxMixedSTT_TST_TTS,

      # syntaxStems2Sen,
      syntaxStems12Sen,
      # syntaxStems2DepSen,
      # syntaxStemsLSen,
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

    self.classifierFs1 = [
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

      # char ngrams
      ngramsLogistic_2,
      ngramsLogistic_3,

      # char ngrams - TF
      ngramsTfLogistic_2,
      ngramsTfLogistic_3,

      ngramsRank_2,
      ngramsRank_3,

      ngramsTfRank_2,
      ngramsTfRank_3,

      # tags 
      tagsLogistic,
      tags12Logistic,
      tags12TfLogistic,

      # word shapes
      wordShapesLogistic,
      wordShapesTfLogistic,

      # lang models
      wordsLm_2,
      wordsLm_3,
      wordsLm_4,

      stemLm_2,
      stemLm_3,
      stemLm_4,

      tagLm_2,
      tagLm_3,
      tagLm_4,

      # subseq features
      stemsSubseq2Logistic_5,
      stemsSubseq2SortedLogistic_5,

      stemsSubseq2Rank_5,
      stemsSubseq2SortedRank_5,

      tagsSubseq2Logistic_5,
      tagsSubseq2SortedLogistic_5,

      # mixed ngrams
      mixedST,
      mixedTS,
      mixedST_TS,

      mixedSST,
      mixedSTS,
      mixedSTT,
      mixedTSS,
      mixedTST,
      mixedTTS,
      mixedSTT_TST_TTS,

      # Syntax features
      syntaxStems2,
      syntaxStems12,
      syntaxStems2Dep,
      syntaxStemsL,
      syntaxStemsR,
      syntaxStemsLR,

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

    self.estimator = ensemble.RandomForestRegressor(n_estimators = n_estimators, compute_importances = True, n_jobs = -1)
    self.k = k
    self.n = n
    self.rootDir = rootDir

  def trainPrepare(self, rows):
    rows, indicies = rows

    pool = multiprocessing.Pool()
    XYs = list(pool.imap_unordered(self._prepare1, self._cachedFolds(rows, indicies)))

    X = numpy.vstack(map(itemgetter(0), XYs))
    Y = numpy.concatenate(map(itemgetter(1), XYs))
    return X, Y

  def _prepare1(self, (k, n, rowsTrain, rowsTest)):
    Xk = numpy.column_stack([self._cachedClassify(f, k, n, rowsTrain, rowsTest) for f in self.classifierFs])
    Yk = numpy.array(self._cachedY(k, n, rowsTest))
    return (Xk, Yk)

  def train(self, rows):
    X, Y = self.trainPrepare(rows)

    rows, indicies = rows
    rowsTrain = rows[indicies]

    print '  fit estimator'
    self.estimator.fit(X, Y)

    print '  train weak classifiers'
    self.classifiers = [f() for f in self.classifierFs]
    for e in self.classifiers:
      e.train(rowsTrain)

    return X, Y

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

  def _cachedFolds(self, rows, indicies):
    for i in range(self.n):
      kfoldFile = os.path.join(self.rootDir, 'k%02d_n%02d.npy' % (self.k, i))
      if os.path.exists(kfoldFile):
        kfold = numpy.load(kfoldFile)
      else:
        kfold = cross_validation.KFold(len(indicies), k = self.k, indices = True, shuffle = True)
        ensureDir(kfoldFile)
        kfold = [(indicies[train], indicies[test]) for train, test in kfold]
        numpy.save(kfoldFile, kfold)

      for k, (train, test) in enumerate(kfold):
        yield (k, i, rows[train], rows[test])

  def _cachedClassify(self, f, k, n, rowsTrain, rowsTest):
    filePath = os.path.join(self.rootDir, '%s_k%02d_n%02d.npy' % (f.func_name, k, n))
    if os.path.exists(filePath):
      X = numpy.load(filePath)
    else:
      tm = datetime.now()

      e = f()
      e.train(rowsTrain)
      X = e.classify(rowsTest)
      numpy.save(filePath, X)
      print 'not found [%s], trained %s' % (filePath, str(datetime.now() - tm))

    return X

  def _cachedY(self, k, n, rowsTest):
    filePath = os.path.join(self.rootDir, 'y_k%02d_n%02d.npy' % (k, n))
    if os.path.exists(filePath):
      Y = numpy.load(filePath)
    else:
      Y = array([float(row.insult) for row in rowsTest])
      numpy.save(filePath, Y)

    return Y


def cmdStackEvalPrepare():
  print 'load train data'
  trainRows = data.loadTrainJson('d/train.toks.csv')
  print 'load test data'
  testRows = data.loadTrainJson('d/test1.toks.csv')

  print 'prepare eval'
  ev = stack.StackEval('d/stack_1')
  ev.prepare(trainRows, testRows)

def cmdStackEval():
  print 'load train data'
  trainRows = data.loadTrainJson('d/train.toks.csv')
  print 'load test data'
  testRows = data.loadTrainJson('d/test1.toks.csv')

  print 'eval'
  ev = StackEval('d/stack_1')
  
  auc, e = ev.eval(lambda: ensemble.ExtraTreesRegressor(n_estimators = 4000, compute_importances = True, n_jobs = -1))
  
  print 'auc: %f' % auc
  print
  print 'features'
  print '\n'.join(map(str, e.feature_importances_))

if __name__ == '__main__':
  import sys
  if 1 < len(sys.argv):
    cmd = sys.argv[1]
    if cmd == 'stack-eval-prepare':
      cmdStackEvalPrepare()
      sys.exit(0)
    elif cmd == 'stack-eval':
      cmdStackEval()
      sys.exit(0)

  print 'invalid arguments'
