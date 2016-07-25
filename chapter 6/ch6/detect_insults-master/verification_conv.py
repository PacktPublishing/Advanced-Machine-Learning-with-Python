import os
import os.path

def convertVerification(inFile, outFile):
  with open(outFile, 'w') as fout:
    with open(inFile) as fin:
      fin.readline()
      fout.write('Date,Comment\n')

      for line in fin:
        id, insult, dt, rest = line.split(',', 3)
        rest = rest.strip('\r\n').strip(',PrivateTest')
        if not rest.startswith('"""') or not rest.endswith('"""'):
          raise "Invalid line [%s]" % line
        fout.write('%s,%s\n' % (dt, rest))
