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
lemmatizer = WordNetLemmatizer()

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
    word_dictionary_2 = {}

    train_labels = []
    iter_word = 0
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
                                    word_dictionary_2[token] = iter_word
                                    iter_word = iter_word + 1

    return (word_dictionary, word_dictionary_2, train_labels)

# def convert_to_tuple_list(word_dictionary):
#     tuple_list = []
#     for key, value in word_dictionary.items():
#         tuple_list.append((key, value))
#     return tuple_list

def convert_to_matrix(word_dictionary, word_dictionary_2):
    # train_matrix = np.zeros(1,len(word_dictionary))
    # train_matrix = [[0 for x in range(len(word_dictionary))] for y in range(1)]
    # i = 0
    # for key, value in word_dictionary.items():
    #     train_matrix[0][i] = value
    #     i = i + 1
    # return train_matrix
    train_matrix = [[0 for x in range(len(word_dictionary))] for y in range(1)]
    with open('spam.csv', 'rb') as spamcsv:
        readCSV = csv.reader(spamcsv)
        iter_mat = 0
        for row in readCSV:
            sentence = row[1]
            train_matrix[iter_mat] = extract_features(sentence, word_dictionary, word_dictionary_2)

    return train_matrix

def extract_features(sentence, word_dictionary, word_dictionary_2):
    vector = np.zeros(len(word_dictionary))

    if(is_ascii(sentence)):
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if(is_ascii(token)):
                if(not is_stopword(token)):
                    if(not is_containing_punctuation(token)):
                        # Add to word dictionary
                        token = lemmatizer.lemmatize(token)
                        if(token in word_dictionary):
                            print("Token : " + str(token))
                            print("Urutan : " + str(word_dictionary_2[token]))
                            print("Frekuensi : " + str(word_dictionary[token]))
                            vector[word_dictionary_2[token]] = word_dictionary[token]
    print("Vector : ")
    print(vector)
    return vector

if __name__ == '__main__':
    (word_dict_frek, word_dict_2, train_labels) = build_word_dictionary()
    train_matrix = convert_to_matrix(word_dict_frek, word_dict_2)
    print(train_matrix)
    # model1 = MultinomialNB()
    model2 = LinearSVC()
    # model1.fit(train_matrix,train_labels)
    model2.fit(train_matrix,train_labels)
