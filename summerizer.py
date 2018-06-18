#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 02:58:20 2018

@author: junaid
"""
import sys
import nltk.data
from nltk import corpus
from nltk.tokenize import RegexpTokenizer
from nltk import stem
import collections


stopwords = set(corpus.stopwords.words('english'))

def stemwords(words,stemmer=None):
	if not stemmer:
		stemmer = stem.SnowballStemmer('english')
                                 
	for i in range(len(words)):
		words[i] = stemmer.stem(words[i])

	return words


def tokenize(text,tokenizer=None):
	if not tokenizer:
		tokenizer = RegexpTokenizer(r'\w+')
	return tokenizer.tokenize(text)


def preprocess_sentence(sentence):
    words = stemwords(tokenize(sentence))
    sentence = [word for word in words if word not in stopwords]
    return nltk.pos_tag(sentence)
    

def preprocess_para(data):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(data)
    return [(preprocess_sentence(sentence),
             sentences.index(sentence)) for sentence in sentences]

def joinlist(l):
    '''Flattens lists to give all the words'''
    if type(l) == type([]):
        x = [joinlist(i) for i in l]
        if type(x[0]) == type("hi"):
            return " ".join(x)
    elif type(l[0]) == type("hi"):
        return l
    
def getTopNwords(paragraphs, N=10):
    word_freq = sorted(collections.Counter(joinlist(paragraphs)).most_common(),
                  key=lambda x: x[1],reverse=True)[:N]
    return [x[0] for x in word_freq]
    
    
def feature_extraction(paragraphs):
    
    features = ["sentence_thematic","sentence position","sentence length",
                "sentence_pos_rel_para","number_of_proper_nouns",
                "number_of_numerals","number_of_named_entities","tfisf",
                "sentence_centroid_similarity"]
    
    topNwords = getTopNwords(paragraphs,10)
    
    for paragraph,index in paragraphs:
        pass
    

def summerize(file):
    text = open(file).read().split()
    paras = [(preprocess_para(para),text.index(para)) for para in text 
             if len(text.strip()) > 0]
    feature_matrix = feature_extraction(paras)
    
    

def main():
    if sys.argc < 1:
        print("Please enter the file to summerize")
        sys.exit(0)
    
    files = sys.argv[1:]
    for file in files:
        print(summerize(file))
    

if __name__ == '__main__':
	main()
    