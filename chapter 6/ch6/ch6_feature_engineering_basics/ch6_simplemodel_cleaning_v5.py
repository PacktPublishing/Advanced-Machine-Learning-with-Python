from bs4 import BeautifulSoup
import re
import pandas as pd
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import StringIO

#we're going to load in our test and training data 

test = pd.read_csv('testtrolls.csv', header=True, names=['y', 'date', 'Comment'])
training = pd.read_csv('trainingtrolls.csv', header=True, names=['y', 'date', 'Comment', 'Usage'])

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

       # the next command uses stopwords from the natural language toolkit corpus. on your first run you won't have the NLTK corpus and you'll see a pretty informative error message.


        words = [w for w in words if not w in stopwords.words("english")]

      
        # as advised, run nltk.download() ... I'd suggest pulling all of the corpus data (via the corpus collection DL)
        # on the second run, remove nltk.download()!  
        
        #we're initialising our lemmatizer... but before we use it, we'll need to use a Part-of-Speech (PoS) tagger (in this case a Treebank tagger) to identify part of speech.

        lemmatizer = WordNetLemmatizer()
        
        tagged = []
        
        for t in words:
            t = t.lower()
            treebank_tag = pos_tag([t])
            tagged.append(treebank_tag)
     
        #this function just translates between the PoS tags used by our Treebank tagger and the WordNet equivalents.     
     
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

        
        postagged = []
        for t in tagged:
            newtag = t[0][0],get_wordnet_pos(t)
            postagged.append(newtag)
     
                     
        lemmatized = []
        for t in postagged:
            lemmatized.append(lemmatizer.lemmatize(t[0], t[1]))
          
        for t in lemmatized:
            t = np.asarray(t)
        
        return(lemmatized)
        
        
    

        
        
# now we'll apply our function to the training and test data.
              
train_x = training["Comment"].apply(cleaner)
test_x = test["Comment"].apply(cleaner)


#Now that we've lemmatized our terms, we're going to do something a little bit different
# the vectorizer we'll use next uses "term frequency inverse document frequency" vectorization. It turns our text strings into vectors.
    
swds = stopwords.words('english')
vect = TfidfVectorizer(analyzer = "word",input="file", ngram_range = (1,3), min_df = 0, stop_words = swds, max_features=5000)

docs_new = [StringIO.StringIO(x) for x in train_x]
tf = vect.fit_transform(docs_new).toarray()

   
# one thing to bear in mind is that this lemmatizer is okay, but it isn't perfect by any means! To demonstrate this, we can output the vocabulary collected by our vectorizer, which produces a neat list of all of the distinct terms in our corpus.
   
#vocab = vect.get_feature_names()
#print(vocab)
   
#You'll notice definite imperfections and one area where we can get significant improvements is through using a better lemmatizer. For the purposes of this example, we'll press on as we are.

#our vectorizer produced a sparse matrix (through ".toarray" above). We can work directly with the dense vectorizer to do some very cool things, for an example, see Mark Needham's blog at http://www.markhneedham.com/blog/2015/02/15/pythonscikit-learn-calculating-tfidf-on-how-i-met-your-mother-transcripts/
#we'll output the sparse matrix to .csv so that we can see what we're working with.

# please note that the output files can be quite large! The "max_features" parameter can be adjusted for convenience if you just want to look at the vectorized feature set itself.

np.savetxt('train_x.csv', tf, delimiter=',')


#we now repeat the process with our test data.

#vect1 = TfidfVectorizer(analyzer = "word",input="file", ngram_range = (1,3), min_df = 0, stop_words = swds)
docs_new1 = [StringIO.StringIO(x) for x in test_x]
tf1 = vect.fit_transform(docs_new1).toarray()

np.savetxt('test_x.csv', tf1, delimiter=',')

