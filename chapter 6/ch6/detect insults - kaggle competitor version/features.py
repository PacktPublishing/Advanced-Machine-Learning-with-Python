import re
import LM
from LM import KneserNeyLM, ngrams
from wordshape import wordShape, capCount

from utils import *

# Features: Row
def getStems1(row):
  return [stem for stem in set([tok[2].lower() for tok in getToks(row)]) if 1 < len(stem)]

def getStemNgrams(n):
  def f(row):
    return [' '.join(ngram) for ngram in ngrams(n, [tok[2].lower() for tok in getToks(row)])]
  return f

def getStemNgramsSorted(n):
  def f(row):
    return [' '.join(sorted(ngram)) for ngram in ngrams(n, [tok[2].lower() for tok in getToks(row)])]
  return f

def getTags(row):
  return [tok[1] for tok in getToks(row)]

def getStems(row):
  return [tok[2].lower() for tok in getToks(row)]

def getTagNgrams(n):
  def f(row):
    return [' '.join(ngram) for ngram in ngrams(n, [tok[1] for tok in getToks(row)])]
  return f

def getTaggedStems(row):
  return set([tok[2].lower() + '_' + tok[1] for tok in getToks(row) if 1 < len(tok[2])])

def getTaggedStemNgrams(n):
  def f(row):
    return [' '.join(ngram) for ngram in ngrams(n, [tok[2].lower() + '_' + tok[1] for tok in getToks(row) if 1 < len(tok[2])])]
  return f

# Features: text
def getWords(text):
  return [tok.lower() for tok in set(tokenize(text)) if 3 <= len(tok)]

def getNgrams(n):
  def f(text):
    return [' '.join(ngram).lower() for ngram in ngrams(n, tokenize(text))]

  return f

def getNgramsSorted(n):
  def f(text):
    return [' '.join(sorted(ngram)).lower() for ngram in ngrams(n, tokenize(text))]

  return f

def getCharNgrams(n):
  def f(text):
    return [ngram for w in tokenize(text) for ngram in ngrams(n, '$' + w.lower() + '$')]

  return f

def wordShapes(text):
  return set((wordShape(tok) for tok in tokenize(text)))

def getToks(row):
  return (tok for sen in row.sentences for tok in sen.tokens)

reTok = re.compile('[a-z0-9]+', re.I)
def tokenize(s):
  return reTok.findall(s)

def countToks(rows, f, pred):
   return count([f(tok) for row in rows for sen in row.sentences for tok in sen if pred(tok)])

def wordsSeq(row):
  return [tok.lower() for tok in tokenize(row.text)]

# Subseq extractors
#   n - max windows size (take subseq from n consequtive words)
def getSubseq2(seqF, n):
  def f(row):
    seq = seqF(row)
    return set(seq + subseq2(n, seq))
  return f

def getSubseq3(seqF, n):
  def f(row):
    seq = seqF(row)
    return set(seq + subseq2(n, seq) + subseq3(n, seq))
  return f

def getSubseq2Sorted(seqF, n):
  def f(row):
    seq = seqF(row)
    return set(seq + subseq2sorted(n, seq))
  return f

def getSubseq3Sorted(seqF, n):
  def f(row):
    seq = seqF(row)
    return set(seq + subseq2sorted(n, seq) + subseq3sorted(n, seq))
  return f

def subseq2(n, xs):
  l = len(xs)
  return ['%s %s' % (xs[i], xs[j]) for i in xrange(l-1) for j in xrange(i+1, i+n+1) if j < l]

def subseq3(n, xs):
  l = len(xs)
  return ['%s %s %s' % (xs[i], xs[j], xs[k]) for i in range(l-2) for j in range(i+1, i+n) for k in range(j+1, i+n+1) if j < l and k < l]

def subseq2sorted(n, xs):
  l = len(xs)
  return [' '.join(sorted((xs[i], xs[j]))) for i in range(l) for j in range(i+1, l) if j - i <= n - 1]

def subseq3sorted(n, xs):
  l = len(xs)
  return [' '.join(sorted((xs[i], xs[j], xs[k]))) for i in range(l) for j in range(i+1, l) for k in range(j+1, l) if k - i <= n - 1]

#
# Mixed ngrams (ex: word0-TAG1, TAG0-word1)

# stem-TAG
def mixedNgramsST(row):
  return ['%s %s' % (ngram[0][2], ngram[1][1]) for ngram in ngrams(2, list(getToks(row)))]

# TAG-stem
def mixedNgramsTS(row):
  return ['%s %s' % (ngram[0][1], ngram[1][2]) for ngram in ngrams(2, list(getToks(row)))]

# stem-stem-TAG
def mixedNgramsSST(row):
  return ['%s %s %s' % (ngram[0][2], ngram[1][2], ngram[2][1]) for ngram in ngrams(3, list(getToks(row)))]

# stem-TAG-stem
def mixedNgramsSTS(row):
  return ['%s %s %s' % (ngram[0][2], ngram[1][1], ngram[2][2]) for ngram in ngrams(3, list(getToks(row)))]

# stem-TAG-TAG
def mixedNgramsSTT(row):
  return ['%s %s %s' % (ngram[0][2], ngram[1][1], ngram[2][1]) for ngram in ngrams(3, list(getToks(row)))]

# TAG-stem-stem
def mixedNgramsTSS(row):
  return ['%s %s %s' % (ngram[0][1], ngram[1][2], ngram[2][2]) for ngram in ngrams(3, list(getToks(row)))]

# TAG-stem-TAG
def mixedNgramsTST(row):
  return ['%s %s %s' % (ngram[0][1], ngram[1][2], ngram[2][1]) for ngram in ngrams(3, list(getToks(row)))]

# TAG-TAG-stem
def mixedNgramsTTS(row):
  return ['%s %s %s' % (ngram[0][1], ngram[1][1], ngram[2][2]) for ngram in ngrams(3, list(getToks(row)))]

# Dependency parsing extractors

# Return dependencies
def getDeps(row):
  return [dep for sen in row.sentences for dep in sen.deps]

def synWords2(row):
  return ['%s %s' % (dep[1].word.lower(), dep[2].word.lower()) for dep in getDeps(row)]

def synStems2(row):
  return ['%s %s' % (dep[1].stem.lower(), dep[2].stem.lower()) for dep in getDeps(row)]

def synStems2Dep(row):
  return ['%s %s %s' % (dep[0], dep[1].stem.lower(), dep[2].stem.lower()) for dep in getDeps(row)]

def synStemsL(row):
  return ['%s %s' % (dep[0], dep[1].stem.lower()) for dep in getDeps(row)]

def synStemsR(row):
  return ['%s %s' % (dep[0], dep[2].stem.lower()) for dep in getDeps(row)]

# using dependency graph
def synToks2(row):
  return [(tok1, tok2) for sen in row.sentences for tok1, tok2 in senToks2(sen)]

def synToks3(row):
  return [(tok1, tok2, tok3) for sen in row.sentences for tok1, tok2, tok3 in senToks3(sen)]

def synWords3(row):
  return ['%s %s %s' % (tok1.word, tok2.word, tok3.word) for tok1, tok2, tok3 in synToks3(row)]

def synStems3(row):
  return ['%s %s %s' % (tok1.stem, tok2.stem, tok3.stem) for tok1, tok2, tok3 in synToks3(row)]

def synTags2(row):
  return ['%s %s' % (tok1.tag, tok2.tag) for tok1, tok2 in synToks2(row)]

def synTags3(row):
  return ['%s %s %s' % (tok1.tag, tok2.tag, tok3.tag) for tok1, tok2, tok3 in synToks3(row)]

def synMixedST(row):
  return ['%s %s' % (tok1.stem, tok2.tag) for tok1, tok2 in synToks2(row)]

def synMixedTS(row):
  return ['%s %s' % (tok1.stem, tok2.tag) for tok1, tok2 in synToks2(row)]

def synMixedSST(row):
  return ['%s %s %s' % (tok1.stem, tok2.stem, tok3.tag) for tok1, tok2, tok3 in synToks3(row)]

def synMixedSTS(row):
  return ['%s %s %s' % (tok1.stem, tok2.tag, tok3.stem) for tok1, tok2, tok3 in synToks3(row)]

def synMixedSTT(row):
  return ['%s %s %s' % (tok1.stem, tok2.tag, tok3.tag) for tok1, tok2, tok3 in synToks3(row)]

def synMixedTSS(row):
  return ['%s %s %s' % (tok1.tag, tok2.stem, tok3.stem) for tok1, tok2, tok3 in synToks3(row)]

def synMixedTST(row):
  return ['%s %s %s' % (tok1.tag, tok2.stem, tok3.tag) for tok1, tok2, tok3 in synToks3(row)]

def synMixedTTS(row):
  return ['%s %s %s' % (tok1.tag, tok2.tag, tok3.stem) for tok1, tok2, tok3 in synToks3(row)]

# sentence -> (tok1, tok2, tok3) - syntax trigrams
def senToks3(sentence):
  toks = senLinks(sentence)
  return [(tok1, tok2, tok3) for tok1 in toks for tok2 in map(itemgetter(1), tok1.outlinks) for tok3 in map(itemgetter(1), tok2.outlinks)]

def senToks2(sentence):
  toks = senLinks(sentence)
  return [(tok1, tok2) for tok1 in toks for tok2 in map(itemgetter(1), tok1.outlinks)]

# sentence -> [(word, tag, stem, [(dep, index)]]
def senLinks(sentence):
  toks = [Syn(word.lower(), tag, stem.lower(), list()) for word, tag, stem in sentence.tokens]
  for typ, gov, dep in sentence.deps:
    if 0 < gov.index:
      toks[dep.index-1].outlinks.append((typ, toks[gov.index-1]))

  return toks

class Syn:
  def __init__(self, word, tag, stem, outlinks):
    self.word = word
    self.tag = tag
    self.stem = stem
    self.outlinks = outlinks

  def __repr__(self):
    return '(%s %s %s [%s])' % (self.word, self.tag, self.stem, ' '.join(['%s,%s' % (typ, syn.word) for typ, syn in self.outlinks]))
