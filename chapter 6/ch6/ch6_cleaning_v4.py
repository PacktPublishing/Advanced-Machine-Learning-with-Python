# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 18:31:04 2016

@author: LegendsUser
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 01:59:23 2016

@author: LegendsUser
"""



from bs4 import BeautifulSoup
import re
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

trolls = pd.read_csv('trolls.csv', header=True, names=['y', 'date', 'Comment'])

# We'll do most of the data cleaning work within one giant function. 
# it's slow, row-based iteration, which costs a few seconds.


def cleaner(inputdata):
    
    lines = []    
    
    for row in inputdata:
        
        
    
        line = BeautifulSoup(str(inputdata), "html.parser")
        
        
        line = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', '_EM', str(line))
        line = re.sub(r'\w+:\/\/\S+', r'_U', line)
        
        # First off we format white space
        line = line.replace('"', ' ')
        line = line.replace('\'', ' ')
        line = line.replace('_', ' ')
        line = line.replace('-', ' ')
        line = line.replace('\n', ' ')
        line = line.replace('\\n', ' ')
        line = line.replace('\'', ' ')
        line = re.sub(' +',' ', line)
        line = line.replace('\'', ' ')
        
        
        # next we kill off any punctuation issues, employing tags (such as _Q, _X) where we feel the punctuation might lead to useful features.
        line = re.sub(r'([^!\?])(\?{2,})(\Z|[^!\?])', r'\1 _BQ\n\3', line)
        line = re.sub(r'([^\.])(\.{2,})', r'\1 _SS\n', line)
        line = re.sub(r'([^!\?])(\?|!){2,}(\Z|[^!\?])', r'\1 _BX\n\3', line)
        line = re.sub(r'([^!\?])\?(\Z|[^!\?])', r'\1 _Q\n\2', line)
        line = re.sub(r'([^!\?])!(\Z|[^!\?])', r'\1 _X\n\2', line)
        line = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2 _EL', line)
        line = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2 _EL', line)
        line = re.sub(r'(\w+)\.(\w+)', r'\1\2', line)
        
        #more encoding. This time we're encoding things like swearing (_SW). Internet trolls can be pretty sweary.

        line = re.sub(r'([#%&\*\$]{2,})(\w*)', r'\1\2 _SW', line)
        line = re.sub('[1|2|3|4|5|6|7|8|9|0]', '', line)
        
        line = re.sub(r' [8x;:=]-?(?:\)|\}|\]|>){2,}', r' _BS', line)
        line = re.sub(r' (?:[;:=]-?[\)\}\]d>])|(?:<3)', r' _S', line)
        line = re.sub(r' [x:=]-?(?:\(|\[|\||\\|/|\{|<){2,}', r' _BF', line)
        line = re.sub(r' [x:=]-?[\(\[\|\\/\{<]', r' _F', line)
        line = re.sub('[%]', '', line)
        
        lines.append(line)
                
        # now we're starting to split the comment into individual 1-word phrases, as an ['foo', 'bar', 'baz'] array instead of "foo bar baz"
               
        phrases = re.split(r'[;:\.()\n]', line)
        phrases = [re.findall(r'[\w%\*&#]+', ph) for ph in phrases]
        phrases = [ph for ph in phrases if ph]
  
        
        words = []
        
        
        
        for ph in phrases:
            words.extend(ph)
        
        tmp = words
        words = []
        new_word = ''
        for word in tmp:
           if len(word) == 1:
              new_word = new_word + word
           else:
              if new_word:
                 words.append(new_word)
                 new_word = ''
              words.append(word)

        words = [w for w in words if not w in stopwords.words("english")]

        # stopwords from the natural language toolkit corpus. on your first run you won't have the NLTK corpus and you'll see a pretty informative error message.

        # as advised, run nltk.download() ... I'd suggest pulling all of the corpus data (via the corpus collection DL)
        # on the second run, remove nltk.download()!  
        

        lemmatizer = WordNetLemmatizer()
        
        tagged = []
        
        for t in words:
            t = t.lower()
            treebank_tag = pos_tag([t])
            tagged.append(treebank_tag)
     
        def get_wordnet_pos(tagged):

            if treebank_tag[0][1].startswith('J'):
                return wordnet.ADJ
            elif treebank_tag[0][1].startswith('V'):
                return wordnet.VERB
            elif treebank_tag[0][1].startswith('N'):
                return wordnet.NOUN
            elif treebank_tag[0][1].startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        
        foo = []
        for t in tagged:
            stuff = t[0][0],get_wordnet_pos(t)
            foo.append(stuff)
            
                     
        tagged = []
        for t in foo:
            tagged.append(lemmatizer.lemmatize(t[0], t[1]))
          
        
        #lemmas = []
        #for t in tags:
        #     lemma = lemmatizer.lemmatize(str(t))
        #     lemmas.append(lemma)
             
        #print(lemmas)
        
        #lemmatizer = WordNetLemmatizer()
        #words = tags.apply(lemmatizer.lemmatize)
        #print(words)

        
        return(tagged)
        
        
        

                
        # next we'll employ a stemmer and lemmatisation.
                
        # you might be wondering why we're doing both. There's a specific type of use case that stemming won't catch
        # that lemmatisation does (e.g. "better" becomes "good"). I ran with both functions here partially to show them off...
                
        # you could survive with just lemmatisation.
       
        #stemmer = PorterStemmer()
        #words = stemmer.stem(str(words))
        
        
        #need some POS tagging here or teh lemmatizer doesn't work???


        
        
# UNLEASH THE TRAINING KRAKEN
              
train_x = trolls["Comment"].apply(cleaner)

vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 3946)
train_y = trolls["y"]
print(train_x)
train_x = vectorizer.fit_transform(train_x[0])
train_x = train_x.toarray()


moretrolls = pd.read_csv('moretrolls.csv', header=True, names=['y', 'date', 'Comment', 'Usage'])

# UNLEASH THE KRAKEN

test_x = moretrolls["Comment"].apply(cleaner)
test_y = moretrolls["y"]
test_x = vectorizer.fit_transform(test_x[0])
test_x = test_x.toarray()

if __name__ == '__main__':

    np.random.seed(0)

    n_folds = 10
    verbose = True
    
    skf = list(StratifiedKFold(train_y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1,  
        criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1,   
            criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, 
            criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, 
            criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, 
            subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((train_x.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((test_x.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((test_x.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = train_x
            y_train = train_y
            X_test = test_x
            y_test = test_y
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[y_test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_test, test_y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min())/(y_submission.max() - y_submission.min())

    print "Saving Results."
    np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')

fpr, tpr, _ = roc_curve(y_test, y_submission)
roc_auc = auc(fpr, tpr)
print("Random Forest benchmark AUC, 1000 estimators")
print(roc_auc)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

        