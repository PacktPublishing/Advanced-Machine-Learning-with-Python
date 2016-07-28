import math
import cPickle

class KneserNeyLM:
  def __init__(self, n, delta = 0.75):
    self.delta = delta
    self.n = n # n-gram level
    # {prefix->(count, {char->count})
    self.counts = dict()
  
  def train(self, texts):
    for seq in texts:
      self.addWord(seq)
    self.calcN1()

  def addWord(self, seq):
    for n in range(2, self.n + 1):
      for ngram in ngrams(n, seq):
        self.addCounts(tuple(ngram[:-1]), ngram[-1])

  def addCounts(self, prefix, ch):
    c, dist = self.counts.setdefault(prefix, (0, dict()))
    dist[ch] = dist.get(ch, 0) + 1
    self.counts[prefix] = (c+1, dist)

  def calcN1(self):
    n1 = dict()
    n1total = 0
    for ch1, dist in [(ch, dist) for (ch, (c, dist)) in self.counts.items() if 1 == len(ch)]:
      for ch2 in dist.keys():
        n1[ch2] = n1.get(ch2, 0) + 1
        n1total += 1
    self.counts[tuple([])] = (n1total, n1)

  def score(self, seq):
    scores = [math.log(1e-10 + self.scoreNgram(tuple(ngram[:-1]), ngram[-1])) for ngram in ngrams(self.n, seq)]
    return sum(scores)

  def scoreNgram(self, prefix, ch):
    # print '%s%s: c(ab)=%i c(a_)=%i' % (prefix, ch, self.getCount(prefix, ch), self.getSumCount(prefix))
    if 0 == len(prefix):
      score = (0.0+self.getCount(prefix, ch))/(0.0+self.getSumCount(prefix))
    else:
      a2 = max(0.0 + self.getCount(prefix, ch) - self.delta, 0.0) / self.getSumCount(prefix)
      a1 = (self.delta/self.getSumCount(prefix)) * (0.0 + self.getNum(prefix)) * (0.0 + self.scoreNgram(prefix[1:], ch))
      # print '  %.5f %.5f' % (a2, a1)
      score = a2 + a1
    # print '  score=%.5f' % (score)
    return score

  def getCount(self, prefix, ch):
    o = self.counts.get(prefix)
    if None == o:
      return 0
    else:
      c, dist = o
      return dist.get(ch, 0)

  def getSumCount(self, prefix):
    return self.counts.get(prefix, (1, {}))[0]

  def getNum(self, prefix):
    o = self.counts.get(prefix)
    if None == o:
      return 0
    else:
      c, dist = o
      return len(dist.keys())

  def save(self, fileName):
    with open(fileName, 'w') as f:
      cPickle.dump(self.counts, f)

  def load(self, fileName):
    with open(fileName, 'r') as f:
      self.counts = cPickle.load(f)

def ngrams(n, s):
  for i in range(0, len(s)-n+1):
    yield s[i:i+n]
