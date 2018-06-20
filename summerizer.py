#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 02:58:20 2018

@author: junaid
"""
import sys
import nltk.data
from nltk import corpus
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk import stem
import collections
from nltk.tag import pos_tag
from nltk import ne_chunk
from nltk.tree import Tree
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


stopwords = set(corpus.stopwords.words('english'))
stemmer = stem.SnowballStemmer('english')

def preprocess_sentence(sentence):
    sentence = word_tokenize(sentence)
    taggedWords = nltk.pos_tag(sentence)
    return [(stemmer.stem(word),ptag) for word,ptag in taggedWords if word.lower() not in stopwords]
    

def preprocess_paragraphs(paragraphs):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    inverse_sentence_structure = dict()
    sentence_number = 0
    processed_paras = list()
    for paragraph in paragraphs:
	    sentences = tokenizer.tokenize(paragraph)
	    sentence_list = list()
	    for sentence in sentences:
	    	inverse_sentence_structure[sentence_number] = sentence
	    	sentence_list.append((preprocess_sentence(sentence)),sentences.index(sentence))
	    processed_paras.append(sentence_list)
	return processed_paras,inverse_sentence_structure

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
    return len(tokenizer.tokenize(" ".join([word for word,_ in sentence])))


def getNumberOfNamedEntities(sentence):
     chunked = ne_chunk(pos_tag(word_tokenize(" ".join([word for word,_ in sentence]))))
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
        	words = list()
        	for word,tag in sentence:
        		words.append(word)
        	l.append(" ".join(words).strip())
    tfidf = v.fit_transform(l)
    vocab = v.vocabulary_
    return tfidf.A,vocab

def get_tfisf_score(sentence,pos,vectors,vocab):
	val = 0.0
	for word,tag in sentence:
		try:
			temp = vectors[pos]
			try:
				
				tf = temp[vocab[word]]
				isf = math.log(vectors.shape[0]/sum(vectors[:,vocab[word]]))
				val += tf*isf
			except:
				pass
		except:
			print(word.strip()," is not in the vocabulary")
	return val

def cosine_sim(s1,s2,tfisf,vocab):
	return np.dot(tfisf[s1], tfisf[s2])


def feature_extraction(paragraphs):
    
	features = ["sentence","sentence position","sentence_thematic",
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
			# print("sentence is", sentence)
			sentence_thematic = number_of_thematic_words(sentence,topNwords)/len(sentence)
			# print("Sentence thematic ",sentence_thematic)
			npropernouns = getNumberOfProperNouns(sentence)
			# print("Number of proper nouns ",npropernouns)
			nnumerals = getNumberOfNumerals(sentence)
			# print("number of numerals ",nnumerals)
			number_of_named_entities = getNumberOfNamedEntities(sentence)
			# print("number of named entities ",number_of_named_entities)
			val = get_tfisf_score(sentence,sentence_pos,tfisf,vocab)
			# print("tsisf score ",val)
			scores.append((val,sentence_pos))
			sentence_length = len(sentence)
			# print(sentence_length)
			sentences_feature_matrix.append([sentence,sentence_pos,sentence_thematic, 
			               sentence_length, sentence_pos_rel_para,
			               npropernouns, nnumerals,
			               number_of_named_entities,val])
			# x = input("Enter to continue")
			sentence_pos += 1

	_, centroid_sentence_index = max(scores)
	sentences_feature_matrix_ = list()
	for sentence_attributes in sentences_feature_matrix:
		l = sentence_attributes
		l.append(cosine_sim(centroid_sentence_index,
		            sentence_attributes[1],tfisf,vocab))
		sentences_feature_matrix_.append(l)

	sentences_feature_matrix = None
	return pd.DataFrame(np.array(sentences_feature_matrix_),columns=features)

def create_yes_no_column(inverse_sentence_structure, outputfile):
	output = open(outputfile).read()
	present = [0]*len(inverse_sentence_structure)
    for index,sentence in inverse_sentence_structure.items():
    	present[index] = sentence in output
    return present

def train_and_plot_results(data):
    pass     

def preprocess_file(file):
	text = open(file).read().split('\n\n')
	paras = [para.replace('\n',' ') for para in text]
	return paras

def summerize(file):
	paragraphs = preprocess_file(file)
	paragraphs,inverse_sentence_structure = preprocess_paragraphs(paragraphs)
	data = feature_extraction(paragraphs)
	present = create_yes_no_column(inverse_sentence_structure, outputfile)
	data['present'] = present
	print(data.head())
	print(data.columns)
	train_and_plot_results(data)
    

summerize('article8')
# def main():
#     if sys.argc < 1:
#         print("Please enter the file to summerize")
#         sys.exit(0)
    
#     files = sys.argv[1:]
#     for file in files:
#         print(summerize(file))
    

# if __name__ == '__main__':
# 	main()