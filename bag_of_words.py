#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation





def review_to_wordlist(review,remove_stopwords):
    review=BeautifulSoup(review).get_text()
    review=re.sub("[^a-zA-Z]"," ",review)
    word=review.lower().split()
    if remove_stopwords:
        stops=set(stopwords.words("english"))
        word=[w for w in word if not w in stops]
    return (" ".join(word))
    
    
    
if __name__=='__main__':    
    train=pd.read_csv(‘data/labeledTrainData.tsv',header=0,delimiter="\t",quoting=3)
    test=pd.read_csv(‘data/testData.tsv',header=0,delimiter="\t",quoting=3)
    
    
    clean_train_reviews=[]
    clean_test_reviews=[]
    for i in range(0,len(train["review"])):
        clean_train_reviews.append(review_to_wordlist(train["review"][i],True))
    for i in range(0,len(test["review"])):
        clean_test_reviews.append(review_to_wordlist(test["review"][i],True))

    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(train_data_features)
    np.asarray(test_data_features)
    
    forest=RandomForestClassifier(n_estimators=100)
    forest=forest.fit(train_data_features,train['sentiment'])
    result=forest.predict(test_data_features)
    
    print ("10 Fold CV Score: ", np.mean(cross_validation.cross_val_score(forest, train_data_features,train['sentiment'], cv=10)))

    
    output=pd.DataFrame(data={"id":test["id"],"sentiment":result})
    output.to_csv(‘result/Bag_of_Words_model.csv',index=False,quoting=3)

    