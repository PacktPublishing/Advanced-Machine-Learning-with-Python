from operator import itemgetter

import os, errno
import os.path

# [a] -> {a:int}
def count(xs):
  d = dict()
  for x in xs:
    d[x] = d.get(x, 0) + 1

  return sorted(d.items(), key = itemgetter(1), reverse = True)

def ensureDir(path):
  mkdir_p(os.path.dirname(path))

def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno == errno.EEXIST:
      pass
    else: raise
