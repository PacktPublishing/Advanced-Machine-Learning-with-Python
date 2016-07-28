Hello!

This content in the associated chapter discusses feature engineering (specifically, a number of text engineering techniques). The code provided here is intended to serve as a guide to that chapter content.

To this end, two sets of solutions to the same problem are presented. One introduces basic concepts in text engineering. The other is a source of ideas for development of more sophisticated solutions.

The first is provided in the subdirectory "ch6_feature_engineering_basics". It provides a single, legible and thoroughly-commented function introducing the feature selection techniques we discussed to the reader. It can be run as a standalone script to produce vectorized text output based on input data. Very simplistic downstream models (an example randomforest and blended ensemble) are provided; these models serve purely as guidance to the interested reader about how cleaning output might be tied into script input.

The content of this chapter does not allow for detailed discussion of how to build a high-performing solution. While the types of models that can be used are referenced, a lot of the very problem-specific feature selection choices are passed over in favour of giving a clear introduction to some involved concepts.

An example of a more highly-performing solution is provided in the subdirectory "detect insults - kaggle competitor version". This is a fork of a code repository provided by the github user Tuzzeg at https://github.com/tuzzeg/detect_insults and the original code within was written entirely by Tuzzeg. It is provided solely so that the interested reader can discover more text feature engineering options. Users interested in understanding advanced text feature engineering configurations should consult the script "features.py" within this directory. These features are combined into a single feature set using the script "stack.py". When run, this code yields an extremely performant result; Tuzzeg used it to place 2nd during the Kaggle contest this data was used in.

In order to run the scripts within this chapter's code repository, please ensure you have the following libraries to hand:

- NLTK
- BeautifulSoup

As well as numpy, sklearn and pandas.

Ensure that you run NLTK.download() to download corpus data (including stopwords).

