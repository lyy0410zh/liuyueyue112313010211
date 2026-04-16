#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kaggle Competition Submission Script
Bag of Words Meets Bags of Popcorn - Movie Review Sentiment Analysis

基于多种特征工程和模型集成的完整解决方案
"""

import os
import re
import logging
import warnings
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import nltk.data

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self):
        self.neg_words = {"not", "never", "no", "none", "hardly", "rarely",
                         "don", "didn", "wasn", "isn", "aren", "wouldn",
                         "couldn", "shouldn", "won", "can't", "cannot"}
        try:
            self.stop_words = set(stopwords.words("english"))
        except:
            from english_stopwords import english_stopwords
            self.stop_words = english_stopwords

    def review_to_wordlist(self, review, remove_stopwords=True):
        """将评论转换为词列表"""
        review_text = BeautifulSoup(review, "html.parser").get_text()
        review_text = re.sub("[^a-zA-Z\'\"]", " ", review_text)
        words = review_text.lower().split()

        if remove_stopwords:
            words = [w for w in words if w not in self.stop_words or w in self.neg_words]

        return words

    def review_to_sentences(self, review):
        """将评论拆分为句子列表"""
        raw_sentences = review.strip().split('. ')
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(self.review_to_wordlist(raw_sentence, remove_stopwords=False))
        return sentences

    def clean_reviews(self, reviews, remove_stopwords=True):
        """批量清理评论"""
        return [" ".join(self.review_to_wordlist(review, remove_stopwords))
                for review in reviews]


class FeatureExtractor:
    """特征提取器"""

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.word2vec_model = None
        self.tfidf_dict = None

    def train_tfidf(self, clean_reviews, max_features=5000, ngram_range=(1, 2)):
        """训练TF-IDF向量化器"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        tfidf_features = self.tfidf_vectorizer.fit_transform(clean_reviews)

        # 构建TF-IDF权重字典（用于Word2Vec加权）
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.tfidf_dict = dict(zip(feature_names, self.tfidf_vectorizer.idf_))

        print(f"TF-IDF model trained with {len(feature_names)} features")
        return tfidf_features

    def train_count_vectorizer(self, clean_reviews, max_features=5000):
        """训练词袋模型"""
        self.count_vectorizer = CountVectorizer(
            analyzer="word",
            tokenizer=None,
            preprocessor=None,
            stop_words=None,
            max_features=max_features
        )
        count_features = self.count_vectorizer.fit_transform(clean_reviews)
        print(f"Count Vectorizer trained with {max_features} features")
        return count_features

    def train_word2vec(self, sentences, num_features=300, min_word_count=10,
                       context=10, num_workers=4):
        """训练Word2Vec模型"""
        print("Training Word2Vec model...")
        self.word2vec_model = Word2Vec(
            sentences,
            workers=num_workers,
            vector_size=num_features,
            min_count=min_word_count,
            window=context,
            sample=1e-3,
            sg=1,
            hs=1,
            epochs=10
        )
        self.word2vec_model.init_sims(replace=True)

        # 测试模型
        try:
            print("Model test - doesn't match:", self.word2vec_model.wv.doesnt_match("man woman child kitchen".split()))
            print("Most similar to 'awful':", self.word2vec_model.wv.most_similar("awful")[:3])
        except Exception as e:
            print(f"Warning: Could not test Word2Vec model: {e}")

        print(f"Word2Vec model trained with {num_features} dimensions")
        return self.word2vec_model

    def make_feature_vec(self, words, use_tfidf_weight=True):
        """为单个文档创建平均词向量"""
        num_features = self.word2vec_model.vector_size
        feature_vec = np.zeros((num_features,), dtype="float32")
        nwords = 0.

        try:
            index2word_set = set(self.word2vec_model.wv.index_to_key)
        except AttributeError:
            index2word_set = set(self.word2vec_model.wv.index2word)

        for word in words:
            if word in index2word_set:
                nwords += 1.
                if use_tfidf_weight and self.tfidf_dict and word in self.tfidf_dict:
                    weight = self.tfidf_dict.get(word, 1.0)
                    feature_vec = np.add(feature_vec, self.word2vec_model.wv[word] * weight)
                else:
                    feature_vec = np.add(feature_vec, self.word2vec_model.wv[word])

        if nwords > 0:
            feature_vec = np.divide(feature_vec, nwords)

        return feature_vec

    def get_avg_feature_vectors(self, reviews_wordlists, use_tfidf_weight=True):
        """批量获取平均词向量"""
        num_features = self.word2vec_model.vector_size
        review_feature_vecs = np.zeros((len(reviews_wordlists), num_features), dtype="float32")

        counter = 0
        for review in reviews_wordlists:
            if counter % 1000 == 0:
                print(f"Processing review {counter}/{len(reviews_wordlists)}")
            review_feature_vecs[counter] = self.make_feature_vec(review, use_tfidf_weight)
            counter += 1

        return review_feature_vecs


class ModelEnsemble:
    """模型集成类"""

    def __init__(self):
        self.models = {}
        self.weights = {}

    def add_model(self, name, model, weight=1.0):
        """添加模型到集成中"""
        self.models[name] = model
        self.weights[name] = weight

    def train_all(self, X_train, y_train):
        """训练所有模型"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            print(f"{name} trained successfully")

    def predict_proba_ensemble(self, X_test):
        """使用加权平均进行概率预测集成"""
        proba_sum = None
        total_weight = 0

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_test)[:, 1]
                weight = self.weights[name]

                if proba_sum is None:
                    proba_sum = proba * weight
                else:
                    proba_sum += proba * weight

                total_weight += weight
                print(f"{name}: prediction completed (weight={weight})")
            except Exception as e:
                print(f"Error predicting with {name}: {e}")

        if total_weight > 0:
            ensemble_proba = proba_sum / total_weight
        else:
            ensemble_proba = None

        return ensemble_proba

    def cross_validate(self, X, y, cv=5):
        """交叉验证评估所有模型"""
        results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
                results[name] = {
                    'mean_auc': scores.mean(),
                    'std_auc': scores.std(),
                    'scores': scores
                }
                print(f"{name}: Mean AUC = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
            except Exception as e:
                print(f"Error cross-validating {name}: {e}")
                results[name] = {'error': str(e)}

        return results


def main():
    """主函数：执行完整的Kaggle比赛流程"""

    print("=" * 80)
    print("Kaggle Competition: Bag of Words Meets Bags of Popcorn")
    print("Movie Review Sentiment Analysis - Complete Solution")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    # 加载数据
    print("\n[1/6] Loading data...")
    train = pd.read_csv(os.path.join(data_dir, 'labeledTrainData.tsv'),
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(data_dir, 'testData.tsv'),
                       header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv(os.path.join(data_dir, 'unlabeledTrainData.tsv'),
                                  header=0, delimiter="\t", quoting=3)

    print(f"Training set: {len(train)} reviews")
    print(f"Test set: {len(test)} reviews")
    print(f"Unlabeled set: {len(unlabeled_train)} reviews")

    # 初始化预处理器
    preprocessor = TextPreprocessor()

    # 文本预处理
    print("\n[2/6] Preprocessing text...")
    print("Cleaning training reviews...")
    clean_train_reviews = preprocessor.clean_reviews(train["review"], remove_stopwords=True)
    print("Cleaning test reviews...")
    clean_test_reviews = preprocessor.clean_reviews(test["review"], remove_stopwords=True)

    # 为Word2Vec准备句子数据
    print("Preparing sentences for Word2Vec...")
    sentences = []
    for review in train["review"]:
        sentences += preprocessor.review_to_sentences(review)
    for review in unlabeled_train["review"]:
        sentences += preprocessor.review_to_sentences(review)
    print(f"Total sentences for Word2Vec: {len(sentences)}")

    # 初始化特征提取器
    extractor = FeatureExtractor(preprocessor)

    # 提取TF-IDF特征
    print("\n[3/6] Extracting TF-IDF features...")
    train_tfidf = extractor.train_tfidf(clean_train_reviews, max_features=5000, ngram_range=(1, 2))
    test_tfidf = extractor.tfidf_vectorizer.transform(clean_test_reviews)
    print(f"TF-IDF training shape: {train_tfidf.shape}")
    print(f"TF-IDF test shape: {test_tfidf.shape}")

    # 提取CountVectorizer特征
    print("\nExtracting Bag-of-Words features...")
    train_bow = extractor.train_count_vectorizer(clean_train_reviews, max_features=5000)
    test_bow = extractor.count_vectorizer.transform(clean_test_reviews)
    print(f"BoW training shape: {train_bow.shape}")
    print(f"BoW test shape: {test_bow.shape}")

    # 训练Word2Vec模型
    print("\n[4/6] Training Word2Vec model...")
    extractor.train_word2vec(sentences, num_features=300, min_word_count=10,
                            context=10, num_workers=4)

    # 获取Word2Vec特征
    print("Creating Word2Vec average vectors...")
    train_wordlists = [preprocessor.review_to_wordlist(review) for review in train["review"]]
    test_wordlists = [preprocessor.review_to_wordlist(review) for review in test["review"]]

    train_w2v = extractor.get_avg_feature_vectors(train_wordlists, use_tfidf_weight=True)
    test_w2v = extractor.get_avg_feature_vectors(test_wordlists, use_tfidf_weight=True)
    print(f"Word2Vec training shape: {train_w2v.shape}")
    print(f"Word2Vec test shape: {test_w2v.shape}")

    # 组合所有特征
    print("\n[5/6] Combining features...")
    from scipy.sparse import hstack, csr_matrix

    train_combined = hstack([train_tfidf, train_bow, csr_matrix(train_w2v)])
    test_combined = hstack([test_tfidf, test_bow, csr_matrix(test_w2v)])
    print(f"Combined training shape: {train_combined.shape}")
    print(f"Combined test shape: {test_combined.shape}")

    # 创建和训练模型集成
    print("\n[6/6] Training ensemble models...")

    ensemble = ModelEnsemble()

    # 添加多个模型
    ensemble.add_model('LogisticRegression_TFIDF',
                      LogisticRegression(C=2.0, max_iter=1000, solver='liblinear', random_state=42),
                      weight=1.5)

    ensemble.add_model('LogisticRegression_Combined',
                      LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42),
                      weight=1.2)

    ensemble.add_model('RandomForest',
                      RandomForestClassifier(n_estimators=200, max_depth=None,
                                           min_samples_split=2, random_state=42,
                                           n_jobs=-1),
                      weight=1.0)

    ensemble.add_model('SGDClassifier',
                      SGDClassifier(loss='log', penalty='l2', alpha=1e-4,
                                   max_iter=1000, tol=1e-3, random_state=42,
                                   n_jobs=-1),
                      weight=0.8)

    # 在组合特征上训练
    print("\nTraining on combined features...")
    ensemble.train_all(train_combined, train["sentiment"])

    # 交叉验证评估
    print("\n" + "="*80)
    print("Cross-validation Results:")
    print("="*80)
    cv_results = ensemble.cross_validate(train_combined, train["sentiment"], cv=5)

    # 预测测试集
    print("\nMaking predictions on test set...")
    test_proba = ensemble.predict_proba_ensemble(test_combined)

    # 将概率转换为类别标签
    threshold = 0.5
    test_predictions = (test_proba >= threshold).astype(int)

    # 保存结果
    output = pd.DataFrame(data={"id": test["id"], "sentiment": test_predictions})
    output_file = os.path.join(data_dir, 'kaggle_submission_ensemble.csv')
    output.to_csv(output_file, index=False, quoting=3)

    print("\n" + "="*80)
    print("SUBMISSION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    print(f"Total predictions: {len(test_predictions)}")
    print(f"Positive sentiment: {sum(test_predictions)} ({sum(test_predictions)/len(test_predictions)*100:.2f}%)")
    print(f"Negative sentiment: {len(test_predictions) - sum(test_predictions)} ({(len(test_predictions) - sum(test_predictions))/len(test_predictions)*100:.2f}%)")

    # 同时保存各个单独模型的预测结果用于分析
    print("\nSaving individual model predictions...")
    for name, model in ensemble.models.items():
        pred = model.predict(test_combined)
        individual_output = pd.DataFrame(data={"id": test["id"], "sentiment": pred})
        individual_file = os.path.join(data_dir, f'submission_{name.lower()}.csv')
        individual_output.to_csv(individual_file, index=False, quoting=3)
        print(f"Saved: {individual_file}")

    print("\n" + "="*80)
    print("All submissions generated successfully!")
    print("="*80)

    return output


if __name__ == '__main__':
    result = main()
