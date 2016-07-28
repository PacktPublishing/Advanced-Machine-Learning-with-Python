import json
from numpy import array
from itertools import izip

class Row:
  def __init__(self, dt, text, sentences, insult = None, rawText = None):
    self.dt = dt
    self.text = text
    self.rawText = rawText
    self.insult = insult
    self.sentences = sentences

class Sentence:
  def __init__(self, tokens, deps):
    self.tokens = tokens
    self.deps = deps

class DepWord:
  def __init__(self, word, stem, index):
    self.word = word
    self.stem = stem
    self.index = index

  def __repr__(self):
    return 'w(%s %s %d)' % (self.word, self.stem, self.index)

def rowToks(row):
  return (tok for sentence in row.sentences for tok in sentence)

def loadTrainJson(fileName):
  rows = []
  with open(fileName, 'r') as f:
    f.readline()
    for line in f:
      insult, jsonStr = line.split(',', 1)
      insult = 1 == int(insult)
      sentences, text, rawText = parseSentences(jsonStr)

      rows.append(Row(None, text, sentences, insult))

  return array(rows)

def loadTestJson(fileName):
  rows = []

  with open(fileName, 'r') as f:
    f.readline()
    for line in f:
      dt, jsonStr = line.split(',', 1)
      sentences, text, rawText = parseSentences(jsonStr)
      rows.append(Row(dt, text, sentences, None, rawText))

  return array(rows)

def parseSentences(jsonStr):
  rowObj = json.loads(jsonStr)
  text = rowObj['text']
  rawText = rowObj['rawText']

  sentences = map(parseSentence, rowObj['sentences'])
  return (sentences, text, rawText)

def parseSentence(senObj):
  toks = [(tok['word'], tok['tag'], tok['stem']) for tok in senObj['tokens']]
  deps = map(parseDep, senObj['dependencies'])
  return Sentence(toks, deps)

def parseDep(depObj):
  return (depObj['rel'], parseDepWord(depObj['gov']), parseDepWord(depObj['dep']))

def parseDepWord(depWordObj):
  return DepWord(depWordObj['word'], depWordObj['stem'], depWordObj['index'])

def joinToks(testFile, testToks, outFile):
  with open(testFile) as f1:
    with open(testToks) as f2:
      with open(outFile, 'w') as fout:
        fin = izip(f1, f2)
        l1, l2 = next(fin)
        fout.write(l1)

        for l1, l2 in fin:
          insult = l1.split(',', 1)[0]
          toks   = l2.split(',', 1)[1]
          fout.write('%s,%s' % (insult, toks))
