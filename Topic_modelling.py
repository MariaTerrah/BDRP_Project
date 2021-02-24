# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 00:35:50 2021

@author: shubh
"""

import findspark
findspark.find()
findspark.init()

import pyspark
import random
sc = pyspark.SparkContext(appName="topic_mod")
print("Initialization successful")

from pathlib import Path
import os

import operator;
import re;


path = Path(__file__).parent / "./extracted_text"

for sample in os.listdir(path):
    input_path = os.path.join(path, sample)

    def preprocess(text):
        '''
        Regular expression for removing all non-letter characters in the file.
        '''
        regex = re.compile('[^a-zA-Z ]')
        '''
        Step 1. 
        Remove the non-letter characters.
        '''
        text = text.map(lambda line: regex.sub('', line))
        
        '''
        Step 2.
        Apply a transformation on the RDD text to obtain an RDD words, where each element is a word from the file.
        '''
        words = text.flatMap(lambda line: line.split(' '))
        
        '''
        Step 3.
        Filter out from the RDD words the words with length 0.
        
        '''
        words = words.filter(lambda word: len(word) > 0)
        
        '''
        Step 4.
        Lowercase all words of the RDD words
        
        '''
        words = words.map(lambda word: word.lower())
        
        '''
        Step 5.
        Returns the RDD words.
        '''
        return words
    
    
    # Reads the novel into a RDD
    geo_doc = sc.textFile(str(input_path))
    
    words = preprocess(geo_doc)
    
    print("Number of words ", words.count())
    
    '''
    Top-100 most frequent words
    '''
    # countByValue() counts the number of occurrences of each word.
    # The list will be sorted in ascending order of number of occurrences
    occurrences = sorted(words.countByValue().items(), key=operator.itemgetter(1))
    
    # Reverse the order
    occurrences.reverse()
    print("Top-100 most frequent words")
    i = 0
    for (w, f) in occurrences:
        i += 1
        if i > 100:
            break
        print(w,"--->",f)
        
    def preprocess(text, stopwords):
        '''
        Regular expression for removing all non-letter characters in the file.
        '''
        regex = re.compile('[^a-zA-Z ]')
        '''
        Step 1. 
        Remove the non-letter characters.
        '''
        text = text.map(lambda line: regex.sub('', line))
        
        '''
        Step 2.
        Obtain the RDD containing the words in the file.
        '''
        words = text.flatMap(lambda line: line.split(" "))
    
        '''
        Step 3.
        Filter out the words with length 0.
        '''
        words = words.filter(lambda word: len(word) > 0)
        
        '''
        Step 4.
        Lowercase all words of the RDD $words$
        '''
        words = words.map(lambda word: word.lower())
        
        '''
        Step 5 Remove stop words.
        '''
        words = words.subtract(stopwords)
        
        
        # Returns the words
        return words
    
    
    # Load the stopwords to an RDD
    stop_path= os.getcwd()
    stopwords = sc.textFile(str(stop_path)+"./stopwords.txt")
    words = preprocess(geo_doc, stopwords)
    print("Number of words after stopword removal ", words.count())
    
    # Count the number of occurrences as before.
    occurrences = sorted(words.countByValue().items(), key=operator.itemgetter(1))
    # Reverse the order
    occurrences.reverse()
    print("Top-100 most frequent words")
    i = 0
    for (w, f) in occurrences:
        i += 1
        if i > 100:
            break
        #print(w, ", ", f)
        
    kvwords = words.map(lambda word : (word, 1))
    print(kvwords.take(100))
    
    occurrences = kvwords.reduceByKey(lambda x, y : x + y)
    occurrences = occurrences.sortBy(lambda x: x[1], ascending=False)
    occurrences.take(100)
    
    occurrences = kvwords.groupByKey()
    occurrences = occurrences.map(lambda x : (x[0], len(x[1])))
    occurrences = occurrences.sortBy(lambda x: x[1], ascending=False)
    occurrences.take(100)
    
    occurrences = sorted(kvwords.countByKey().items(), key=operator.itemgetter(1))
    # Reverse the order
    occurrences.reverse()
    print("Top-100 most frequent words")
    i = 0
    for (w, f) in occurrences:
        i += 1
        if i > 100:
            break
        print(w, ", ", f)
        
