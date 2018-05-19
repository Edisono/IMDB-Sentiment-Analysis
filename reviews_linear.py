import os
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation



def review_to_wordlist(review,remove_stopwords):
    review=BeautifulSoup(review).get_text()
    review=re.sub("[^a-zA-Z]"," ",review)
    word=review.lower().split()
    if remove_stopwords:
        stops=set(stopwords.words("english"))
        word=[w for w in word if not w in stops]
    return (" ".join(word))


train = pd.read_csv(‘data/labeledTrainData.tsv', header=0, \
                delimiter="\t", quoting=3)
test = pd.read_csv(‘data/testData.tsv', header=0, delimiter="\t", \
               quoting=3 )
y = train["sentiment"]  


traindata = []
for i in range( 0, len(train["review"])):
    traindata.append(" ".join(review_to_wordlist(train["review"][i], False)))
testdata = []
for i in range(0,len(test["review"])):
    testdata.append(" ".join(review_to_wordlist(test["review"][i], False)))



tfv = TfidfVectorizer(min_df=3,
        strip_accents='unicode',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), sublinear_tf=1,
        stop_words = 'english')
X_all = traindata + testdata
lentrain = len(traindata)

tfv.fit(X_all)
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]


forest=RandomForestClassifier(n_estimators=100)
forest=forest.fit (X,y)
result=forest.predict(X_test)


print ("10 Fold CV Score: ", np.mean(cross_validation.cross_val_score(forest, X, y, cv=10)))

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv(‘result/tfidf_model.csv', index=False, quoting=3)
