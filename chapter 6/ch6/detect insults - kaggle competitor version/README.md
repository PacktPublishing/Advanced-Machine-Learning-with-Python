# Kaggle competetion: Detecting insults

Here I describe my approach for solving [Kaggle competetion](http://www.kaggle.com/c/detecting-insults-in-social-commentary) task. My solution landed at the #2 place.

## General approach

I decided to build a bunch of simple (basic) classifiers and combine them to a final ensemble. As a basic classifiers I used mostly Logisitc regression over words/stemmed words/POS tags/etc. Basic classifiers results were stacked using Random Forest regressor which were producing the final score.

## Basic classifiers

### Word/stem/POS tag models
The most obvious NLP features are words and stems. [Stanford POS tagger](http://nlp.stanford.edu/software/tagger.shtml) were used to retrieve word's POS tags which then were used as an ordinary words/stems features.

<pre>
          0         1         2        3        4          5        6        7
words:    I         really    do       n't      understand your     point    .
stems:    I         realli    do       n't      understand your     point    .
POS tags: PRP       RB        VBP      RB       VB         PRP$     NN       .
</pre>

Using word/stem/POS tag sequence one can build more complicated features: bigrams, trigrams, unordered bigrams/trigrams, subsequences of sliding window of length N and so on.

For example, POS tag based features:
<pre>
                   0         1         2        3        4        5        6
sequence:          PRP       RB        VBP      RB       VB       PRP$     NN
bigrams:           PRP-RB    RB-VBP    VBP-RB   RB-VB    VB-PRP$  PRP$-NN
unordered bigrams: PRP-RB    RB-VBP    RB-VBP   RB-Vb    PRP$-VB  NN-PRP$
2-subseq of 3:     [ PRP,RB,VBP                ]
                     PRP-RB,PRP-VBP,RB-VBP
                             [ RB,VBP,RB                ]
                               RB-VBP,RB-RB,VBP-RB
                   ...
</pre>

After extracting all featurs they should be scored. There are a bunch of different scoring approaches: 0/1, TF, TF\*IDF, Probability score, Mutual information ad so on.
Brief explanation:

* 0/1 - word present/not present in document
* TF  - term frequncy - how many times this word occured in the document
* [TF*IDF](http://en.wikipedia.org/wiki/Tf%E2%80%93idf) - use IDF (inverted document frequncy) to lower weight of popular words
* Probability score - Probability of seeing this word in document of some class
* [Mutual information](http://en.wikipedia.org/wiki/Pointwise_mutual_information)

I used probability score for the most features. 

### Language models
Using positive/negative samples one can build a [Language model](http://en.wikipedia.org/wiki/Language_model) to calculate probability of some document being "insult" or "not insult". I used [Kneser Ney language model](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=479394) over words/stems/POS tags up to 7-grams (however 7-grams language models were not useful at all).

### Mixed (stem + POS tags) models
To add smoothing I tried to build a stem-POS mixed bigrams and trigrams.

<pre>
                   0          1          2        3             4               5          6
words:             I          really     do       n't           understand      your       point
tags:              PRP        RB         VBP      RB            VB              PRP$       NN
tag-word:          PRP-really RB-do      VBP-n't  RB-understand VB-your         PRP$-point
word-tag:          I-RB       really-VBP do-RB    n't-VB        understand-PRP$ your-NN
</pre>

### Character ngrams
Character ngrams based classifiers were used as well - only 2,3,4 grams.

### Syntax features
[Stanford parser](http://nlp.stanford.edu/software/lex-parser.shtml) was used to obtain dependency parsing results and use it as an additional features.
Dependency parser gives a number of triples (dependency type, word1, word2). Using these triples I can build "syntax bigrams" and "syntax trigrams". There also have been extractor for (dependency type, word2) and (word1, dependency type) features.

<pre>
                                bigrams             (dep, w2)       (w1, dep)
nsubj(understand-5, I-1)        "understand I"      NSUBJ-I         understand-NSUBJ
advmod(understand-5, really-2)  "understand really" ADVMOD-really   understand-ADVMOD
aux(understand-5, do-3)         "understand do"     AUX-do          understand-AUX
neg(understand-5, n't-4)        "understand n't"    NEG-n't         understand-NEG
root(ROOT-0, understand-5)      "ROOT understand"   ROOT-understand understand-ROOT
poss(point-7, your-6)           "point your"        POSS-your       point-POSS
dobj(understand-5, point-7)     "understand point"  DOBJ-point      understand-DOBJ
</pre>

In one of my experiments I used only syntax features - and got pretty good AUC. In my final ensemble they had very low feature importance though.

## Metamodels
Two metamodels were implemented (metamodel uses underlying basic model to build another model) - "sentence level metamodels" and "ranking models".

### Sentence level models
The intuition is simple - in insulting message one sentence is usually insulting while others could be just a "normal".

So only examples that contain 1 sentence were selected and underlying model was trained using only those examples. To predict the score message is splitted to sentences and maximum of "insult" score (given by underlying model) is taken.

### Using ranking instead of classifying
For AUC we really care about correct relative scores of predictions, and not about absolute values. Classification problem could be turned into ranking problem.

## Implementation
I've used [Stanford POS tagger](http://nlp.stanford.edu/software/tagger.shtml) and [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) for feature extraction and [scikit-learn](http://scikit-learn.org) package for machine learning. Final model contained 110 basic classifiers and took about 10 hours to train (just because I didn't optimize for speed it at all).

## Experiment results
Here is the top of the most important basic classifiers according to RandomForest feature importances in final submission:

<pre>
stemsSubseq3Logistic_6       0.22
stemsSubseq2SortedLogistic_5 0.16
stemsSubseq3Logistic_5       0.12
stem12Rank                   0.08
stemsSubseq2Logistic_6       0.05
</pre>

Here is an example of using sentence level model with language model inside.
<pre>
stemLm_2    0.8866 0.7902
stemLm_3    0.0410 0.0262
stemLm_4    0.0123 0.0043
stemLm_5    0.0602 0.0344
stemLmSen_2        0.1254
stemLmSen_3        0.0124
stemLmSen_4        0.0038
stemLmSen_5        0.0034

AUC          0.72   0.77
</pre>

So we can see that sentence level features give additional information to the final model.

Here is the feature group importances in the final submission:
<pre>
stem subsequence based         0.66
stem based (unigrams, bigrams) 0.18
char ngrams based (sentence)   0.07
char ngrams based              0.04
all syntax                     0.006
all language models            0.004
all mixed                      0.002
</pre>
