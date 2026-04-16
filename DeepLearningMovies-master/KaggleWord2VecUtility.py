#!/usr/bin/env python

import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup


class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=True ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. 移除HTML标签
        review_text = BeautifulSoup(review, "html.parser").get_text()
        #
        # 2. 保留否定词缩写+英文单词，正则优化
        review_text = re.sub("[^a-zA-Z\'\"]", " ", review_text)
        #
        # 3. 分词+转小写
        words = review_text.lower().split()
        #
        # 4. 定义否定词集合（核心！）
        neg_words = {"not", "never", "no", "none", "hardly", "rarely", "don", "didn", "wasn", "isn", "aren", "wouldn"}
        #
        # 5. 过滤停用词，但保留否定词
        if remove_stopwords:
            from nltk.corpus import stopwords
            stops = set(stopwords.words("english"))
            words = [w for w in words if w not in stops or w in neg_words]
        #
        # 6. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer=None, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Split the paragraph into sentences using simple punctuation
        raw_sentences = review.strip().split('. ')
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
