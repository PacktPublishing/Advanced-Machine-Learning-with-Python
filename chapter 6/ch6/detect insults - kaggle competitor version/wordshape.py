cap_tr = { 'start:a': 'lower',
           'start:A': 'upper',
           'lower:a': 'lower',
           'lower:A': 'mixed',
           'upper:a': 'cap',
           'upper:A': 'upper',
           'mixed:a': 'mixed',
           'mixed:A': 'mixed',
           'cap:a': 'cap',
           'cap:A': 'mixed' }
cap_f  = { 'start': 'o',
           'lower': 'lower',
           'upper': 'o',
           'mixed': 'o',
           'cap': 'cap' }
def capitalized(s):
  def move(state, ch):
    if 'a' <= ch and ch <= 'z':
      return cap_tr[state + ':a']
    else:
      return cap_tr[state + ':A']
  return cap_f[reduce(move, s, 'start')]

def shapeChar(ch):
  if 'a' <= ch and ch <= 'z':
    return 'a'
  elif 'A' <= ch and ch <= 'Z':
    return 'A'
  elif '0' <= ch and ch <= '9':
    return '0'
  else:
    return '#'

shapeMasks = {'A':1, 'a':2, '0':4, '#':8}
def shapeMask(mask, ch):
  return mask | shapeMasks[shapeChar(ch)]

shapeMaskStr = {0:'', 1:'A', 2:'a', 3:'Aa', 4:'0', 5:'A0', 6:'a0', 7:'Aa0', 8:'#', 9:'A#', 10:'a#', 11:'Aa#', 12:'0#', 13:'A0#', 14:'a0#', 15:'Aa0#'}

def wordShape(s):
  return ''.join(map(shapeChar, s[:2])) + shapeMaskStr[reduce(shapeMask, s[2:], 0)]

# Calculate count of characters of classes:
# [A-Z],[a-z],[0-9],[!?],punct
def capCount(s):
  def f((u,l,d,e,o), ch):
    if 'A' <= ch and ch <= 'Z':
      return (u+1,l,d,e,o)
    elif 'a' <= ch and ch <= 'z':
      return (u,l+1,d,e,o)
    elif '0' <= ch and ch <= '9':
      return (u,l,d+1,e,o)
    elif '!' == ch or '?' == ch:
      return (u,l,d,e+1,o)
    else:
      return (u,l,d,e,o+1)
  return reduce(f, s, (0, 0, 0, 0, 0))
