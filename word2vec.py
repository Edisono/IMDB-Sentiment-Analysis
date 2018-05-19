 import pandas as pd
import numpy as np  
import os
import re
from nltk.corpus import stopwords
import nltk
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier


def review_to_wordlist( review, remove_stopwords=False ):

    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

def review_to_sentences( review, tokenizer, remove_stopwords=False ):

    raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordl( raw_sentence,remove_stopwords ))

    return sentences

def makeFeatureVec(words, model, num_features):

    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    for review in reviews:
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



if __name__ == '__main__':

    train = pd.read_csv( 'data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv('data/testData.tsv', header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv('data/unlabeledTrainData.tsv', header=0,  delimiter="\t", quoting=3 )


    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []  

    print "Parsing sentences from training set"
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    print "Parsing sentences from unlabeled set"
    for review in unlabeled_train["review"]:
        sentences +=review_to_sentences(review, tokenizer)

    num_features = 300    
    min_word_count = 40   
    num_workers = 10       
    context = 10         
    downsampling = 1e-3   

    print "Training Word2Vec model..."
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)



    trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features )
    testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features )

    forest = RandomForestClassifier( n_estimators = 100 )

    forest = forest.fit( trainDataVecs, train["sentiment"] )

    result = forest.predict( testDataVecs )

    print ("10 Fold CV Score: ", np.mean(cross_validation.cross_val_score(forest, trainDataVecs,train['sentiment'], cv=10)))


    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( "result/Word2Vec_AverageVectors.csv", index=False, quoting=3 )
