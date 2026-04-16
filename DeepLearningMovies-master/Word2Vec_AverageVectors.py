#!/usr/bin/env python

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Parts 2 and 3 of the tutorial, which cover how to
#  train a model using Word2Vec.
#
# *************************************** #


# ****** Read the two training sets and the test set
#
import pandas as pd
import os
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from KaggleWord2VecUtility import KaggleWord2VecUtility


# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features, tfidf_dict=None):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    try:
        # Gensim 4.0.0+
        index2word_set = set(model.wv.index_to_key)
    except AttributeError:
        # Gensim 3.x
        index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            if tfidf_dict is not None and word in tfidf_dict:
                # 使用TF-IDF权重
                weight = tfidf_dict[word]
                featureVec = np.add(featureVec, model.wv[word] * weight)
            else:
                # 简单平均
                featureVec = np.add(featureVec, model.wv[word])
    #
    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features, tfidf_dict=None):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features, tfidf_dict)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



if __name__ == '__main__':

    # Read data from files
    train = pd.read_csv( os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )

    # Verify the number of reviews that were read (100,000 in total)
    print("Read %d labeled train reviews, %d labeled test reviews, " \
     "and %d unlabeled reviews\n" % (train["review"].size,
     test["review"].size, unlabeled_train["review"].size ))



    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    print("Parsing sentences from training set")
    for review in train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review)

    print("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review)

    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 200    # 词向量维度（降低：300→200，减少过拟合）
    min_word_count = 20   # 最小词频（提升：10→20，过滤更多噪声）
    num_workers = 4       # 多线程加速
    context = 8           # 上下文窗口（降低：10→8，更聚焦）
    downsampling = 1e-3   # 高频词下采样
    epochs = 8            # 训练轮数（降低：10→8，防止过拟合）

    # Initialize and train the model (this will take some time)
    print("Training Word2Vec model...")
    model = Word2Vec(sentences,
                  workers=num_workers,
                  vector_size=num_features,  # 注意：gensim新版本用vector_size，旧版size
                  min_count=min_word_count,
                  window=context,
                  sample=downsampling,
                  sg=1,     # Skip-gram（对低频词/情感词效果更好）
                  hs=1,     # 层次softmax
                  epochs=10)# 训练轮数

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)

    # Test the model
    try:
        print(model.wv.doesnt_match("man woman child kitchen".split()))
        print(model.wv.doesnt_match("france england germany berlin".split()))
        print(model.wv.doesnt_match("paris berlin london austria".split()))
        print(model.wv.most_similar("man"))
        print(model.wv.most_similar("queen"))
        print(model.wv.most_similar("awful"))
    except Exception as e:
        print(f"Error testing model: {e}")

    # ****** Train TF-IDF model for weighted average
    #
    print("Training TF-IDF model...")
    # 准备用于TF-IDF训练的语料
    clean_train_reviews = getCleanReviews(train)
    # 训练TF-IDF模型
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    tfidf.fit(clean_train_reviews)
    # 构建词-权重字典
    tfidf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    print(f"TF-IDF model trained with {len(tfidf_dict)} features")


    
    # ****** Create average vectors for the training and test sets
    #
    print("Creating TF-IDF weighted average feature vecs for training reviews")

    trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features, tfidf_dict )

    print("Creating TF-IDF weighted average feature vecs for test reviews")

    testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features, tfidf_dict )


    # ****** Feature scaling to improve generalization
    print("Scaling features...")
    scaler = StandardScaler()
    trainDataVecs_scaled = scaler.fit_transform(trainDataVecs)
    testDataVecs_scaled = scaler.transform(testDataVecs)

    # ****** Fit a logistic regression to the training set, then make predictions
    #
    # 强正则化逻辑回归（防止过拟合）
    from sklearn.model_selection import cross_val_score
    lr = LogisticRegression(
        C=0.5,            # 降低正则化强度（2.0→0.5），增强正则化
        max_iter=2000,    # 确保收敛
        solver='lbfgs',   # 更适合概率预测
        random_state=42,
        class_weight='balanced'  # 平衡类别权重
    )

    # 交叉验证AUC（比单次划分更准确）
    print("Performing 5-fold cross-validation for AUC...")
    auc_scores = cross_val_score(lr, trainDataVecs_scaled, train["sentiment"], cv=5, scoring='roc_auc')
    print(f"5折交叉验证AUC: {auc_scores.mean():.4f}")

    print("Fitting a logistic regression to labeled training data...")
    lr = lr.fit( trainDataVecs_scaled, train["sentiment"] )

    # Test & extract results - 使用概率预测而非硬分类
    result_proba = lr.predict_proba( testDataVecs_scaled )[:, 1]  # 获取正类概率

    # Write the test results - 使用概率值而非0/1
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result_proba} )
    output.to_csv( os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', "Word2Vec_TFIDF_Weighted_LogisticRegression_v2.csv"), index=False, quoting=3 )
    print("Wrote Word2Vec_TFIDF_Weighted_LogisticRegression_v2.csv")
    print(f"Prediction statistics: min={result_proba.min():.4f}, max={result_proba.max():.4f}, mean={result_proba.mean():.4f}")
