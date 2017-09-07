import sys
import json
import csv
import nltk
import numpy as np
from sklearn import datasets
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix   

stopwords_dict = {}
punctuation_dict = {}

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def is_stopword(s):
    return s in stopwords_dict

def is_containing_punctuation(s):
    for key in punctuation_dict:
        if(key in s):
            return True
    return False

def build_word_dictionary():
    file = open('stopwords-en.json','r+')
    stopwords_list = []
    stopwords_list = json.load(file)

    file.close()

    for stopword in stopwords_list:
        stopwords_dict[stopword] = True

    file = open('punctuation-en.json','r+')
    punctuation_list = []
    punctuation_list = json.load(file)
    file.close()

    for punctuation in punctuation_list:
        punctuation_dict[punctuation] = True

    # Build word dictionary
    word_dictionary = {}
    lemmatizer = WordNetLemmatizer()
    train_labels = []
    with open('spam.csv', 'rb') as spamcsv:
        readCSV = csv.reader(spamcsv)
        for row in readCSV:
            sentence = row[1]
            spam = row[0] # spam = 0, not spam = 1
            if spam == "spam":
                train_labels.append(0)
            else:
                train_labels.append(1)                
            if(is_ascii(sentence)):
                tokens = nltk.word_tokenize(sentence)  
                for token in tokens:
                    if(is_ascii(token)):
                        if(not is_stopword(token)):
                            if(not is_containing_punctuation(token)):
                                # Add to word dictionary
                                token = lemmatizer.lemmatize(token)
                                if(token in word_dictionary):
                                    word_dictionary[token] = word_dictionary[token] + 1
                                else:
                                    word_dictionary[token] = 1
    return (word_dictionary,train_labels)

if __name__ == '__main__':
    # print(build_word_dictionary())
    (train_matrix,train_labels) = build_word_dictionary()
    model1 = MultinomialNB()
    model2 = LinearSVC()
    print type(train_matrix)
    print(train_matrix)
    # model1.fit(train_matrix,train_labels)
    # model2.fit(train_matrix,train_labels)
