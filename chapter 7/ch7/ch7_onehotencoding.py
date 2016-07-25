from sklearn import preprocessing

enc = preprocessing.OneHotEncoder(categorical_features='all', dtype= 'float', handle_unknown='error', n_values='auto', sparse=True)

tweets.delayencode = enc.transform(tweets.location).toarray()
