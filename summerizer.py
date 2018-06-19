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
    pass

def feature_extraction(paragraphs):
    
    features = ["sentence_thematic","sentence position","sentence length",
                "sentence_pos_rel_para","number_of_proper_nouns",
                "number_of_numerals","number_of_named_entities","tfisf",
                "sentence_centroid_similarity"]
    
    topNwords = getTopNwords(paragraphs,10)
    indices = create_indices(paragraphs)
    sentence_feature_matrix = list()
    sentence_pos = 0
    for paragraph,para_index in paragraphs:
        for sentence,sentence_pos_rel_para in paragraph:
            sentence_pos += 1
            sentence_thematic = number_of_thematic_words(sentence,topNwords)/len(sentence)
            npropernouns = getNumberOfProperNouns(sentence)
            nnumerals = getNumberOfNumerals(sentence)
            number_of_named_entities = getNumberOfNamedEntities(sentence)

def summerize(file):
    text = open(file).read().split()
    paras = [(preprocess_para(para,text.index(para)),text.index(para)) for para in text 
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
    