from datetime import datetime
import os
import os.path

import data
from stack import StackModel

def cmdCombine(file1, file2, outFile):
  with open(outFile, 'w') as fout:
    with open(file1) as fin:
      fout.write(fin.readline()) # header
      fout.writelines(fin)

    with open(file2) as fin:
      fin.readline() # skip header
      fout.writelines(fin)

def cmdMakeSubmission(n_estimators, k, n, trainFile, testFile, outFile):
  dt = datetime.now()
  print 'load train data... ',
  trainRows = data.loadTrainJson(trainFile)
  print str(datetime.now() - dt)

  dt = datetime.now()
  print 'load test data... ',
  testRows = data.loadTestJson(testFile)
  print str(datetime.now() - dt)

  dt = datetime.now()
  print 'train model'
  m = StackModel(n_estimators, k, n)
  m.train(trainRows)
  print 'train model, done in %s' % str(datetime.now() - dt)

  m.estimator.n_jobs = 1

  dt = datetime.now()
  print 'generate submission'
  with open(outFile, 'w') as f:
    f.write('Insult,Date,Comment\n')
    f.writelines(('%f,%s,%s\n' % (m.classify1(row), row.dt, row.rawText) for row in testRows))

  print 'generate submission, done in %s' % str(datetime.now() - dt)

if __name__ == '__main__':
  import sys
  if 1 < len(sys.argv):
    cmd = sys.argv[1]
    if cmd == 'combine':
      cmdCombine(sys.argv[2], sys.argv[3], sys.argv[4])
      sys.exit(0)
    elif cmd == 'mksub':
      cmdMakeSubmission(50, 4, 1, sys.argv[2], sys.argv[3], sys.argv[4])
      sys.exit(0)
    elif cmd == 'final_1000_k4_n2':
      cmdMakeSubmission(1000, 4, 2, sys.argv[2], sys.argv[3], sys.argv[4])
      sys.exit(0)
    elif cmd == 'final_2000_k4_n2':
      cmdMakeSubmission(2000, 4, 2, sys.argv[2], sys.argv[3], sys.argv[4])
      sys.exit(0)
    elif cmd == 'final_1000_k4_n4':
      cmdMakeSubmission(1000, 4, 4, sys.argv[2], sys.argv[3], sys.argv[4])
      sys.exit(0)
    elif cmd == 'final_2000_k4_n4':
      cmdMakeSubmission(2000, 4, 4, sys.argv[2], sys.argv[3], sys.argv[4])
      sys.exit(0)
    elif cmd == 'final_4000_k4_n4':
      cmdMakeSubmission(4000, 4, 4, sys.argv[2], sys.argv[3], sys.argv[4])
      sys.exit(0)

  print "invalid arguments"
