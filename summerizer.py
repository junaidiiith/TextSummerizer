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
from nltk.tag import pos_tag,ne_chunk
from nltk.tree import Tree
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


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
    wordlist = list()
    for para,index in l:
        for sentences,ind in para:
            wordlist += sentences
    return wordlist
    
def getTopNwords(paragraphs, N=10):
    word_freq = sorted(collections.Counter(joinlist(paragraphs)).most_common(),
                  key=lambda x: x[1],reverse=True)[:N]
    return [x[0] for x in word_freq]
    

def number_of_thematic_words(sentence, topwords):
    count = 0
    for word in sentence:
        if word in topwords:
            count += 1
    return count

def getNumberOfProperNouns(sentence):
    count = 0
    for word in sentence:
        if word[1] == 'NNP':
            count += 1
    
    return count

def getNumberOfNumerals(sentence):
    tokenizer = RegexpTokenizer(r'\d+.\d+')
    return len(tokenizer.tokenize(sentence))


def getNumberOfNamedEntities(sentence):
     chunked = ne_chunk(pos_tag(tokenize(sentence)))
     continuous_chunk = []
     current_chunk = []
     for i in chunked:
             if type(i) == Tree:
                     current_chunk.append(" ".join([token for token, pos in i.leaves()]))
             elif current_chunk:
                     named_entity = " ".join(current_chunk)
                     if named_entity not in continuous_chunk:
                             continuous_chunk.append(named_entity)
                             current_chunk = []
             else:
                     continue
     return len(continuous_chunk)

def create_indices(paragraphs):
    v = TfidfVectorizer()
    l = list()
    for paragraph,paragraph_index in paragraphs:
        for sentence, sentence_index in paragraph:
            l.append(sentence)
    
    tfidf = v.fit_transform(l)
    vocab = v.vocabulary_
    return tfidf.A,vocab

def get_tfisf_score(sentence,vectors,vocab):
    val = 0.0
    for word in sentence:
        tf = vectors[sentence][vocab[word]]
        isf = math.log(vectors.shape[0]/sum(vectors[:,vocab[word]]))
        val += tf*isf
    return val

def cosine_sim(s1,s2,tfisf,vocab):
    return np.dot(tfisf[vocab[s1]], tfisf(vocab[s2]))


def feature_extraction(paragraphs):
    
    features = ["sentence","sentence_thematic","sentence position",
                "sentence length","sentence_pos_rel_para",
                "number_of_proper_nouns",
                "number_of_numerals","number_of_named_entities","tfisf",
                "sentence_centroid_similarity"]
    
    topNwords = getTopNwords(paragraphs,10)
    tfisf,vocab = create_indices(paragraphs)
    sentences_feature_matrix = list()
    sentence_pos = 0
    scores = list()
    for paragraph,para_index in paragraphs:
        for sentence,sentence_pos_rel_para in paragraph:
            sentence_pos += 1
            sentence_thematic = number_of_thematic_words(sentence,topNwords)/len(sentence)
            npropernouns = getNumberOfProperNouns(sentence)
            nnumerals = getNumberOfNumerals(sentence)
            number_of_named_entities = getNumberOfNamedEntities(sentence)
            tfisf = get_tfisf_score(sentence,tfisf,vocab)
            scores.append((tfisf,sentence))
            sentence_length = sum(len(word) for word in sentence)
            sentences_feature_matrix = [sentence,sentence_pos,sentence_thematic, 
                                   sentence_length, sentence_pos_rel_para,
                                   npropernouns, nnumerals,
                                   number_of_named_entities,tfisf]
            
    _, centroid_sentence = max(scores)
    sentences_feature_matrix_ = list()
    for sentence_attributes in sentences_feature_matrix:
        l = sentence_attributes
        l.append(cosine_sim(centroid_sentence,
                            sentence_attributes[0],tfisf,vocab))
        sentences_feature_matrix_.append(l)
    
    sentences_feature_matrix = None
    return pd.DataFrame(np.array(sentences_feature_matrix_),columns=features)


def train_and_plot_results(data):
    pass     

def summerize(file):
    text = open(file).read().split()
    paras = [(preprocess_para(para,text.index(para)),text.index(para)) for para in text 
             if len(text.strip()) > 0]
    
    data = feature_extraction(paras)
    train_and_plot_results(data)
    

def main():
    if sys.argc < 1:
        print("Please enter the file to summerize")
        sys.exit(0)
    
    files = sys.argv[1:]
    for file in files:
        print(summerize(file))
    

if __name__ == '__main__':
	main()
    