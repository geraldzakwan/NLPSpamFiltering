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
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

stopwords_dict = {}
punctuation_dict = {}
lemmatizer = WordNetLemmatizer()
csv_name = "output.csv"

def len_file():
    with open(csv_name,"r") as f:
        reader = csv.reader(f,delimiter = ",")
        data = list(reader)
        row_count = len(data)
    return row_count

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
    with open(csv_name, 'rb') as spamcsv:
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
    train_matrix = [[0 for x in range(len(word_dictionary))] for y in range(len_file())]
    with open(csv_name, 'rb') as spamcsv:
        readCSV = csv.reader(spamcsv)
        iter_mat = 0
        for row in readCSV:
            sentence = row[1]
            vector = extract_features(sentence, word_dictionary, word_dictionary_2)
            # print("Vector : ")
            # print(vector)
            for x in range(0, len(word_dictionary)):
                train_matrix[iter_mat][x] = vector[x]
            iter_mat = iter_mat + 1

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
                            # print("Token : " + str(token))
                            # print("Urutan : " + str(word_dictionary_2[token]))
                            # print("Frekuensi : " + str(word_dictionary[token]))
                            vector[word_dictionary_2[token]] = word_dictionary[token]
    return vector

def accuracy_split(model, train_matrix, train_labels, train_percentage):
    split_number = train_percentage * len(train_matrix) / 100
    train_matrix_selected = train_matrix[:split_number]
    train_label_selected = train_labels[:split_number]

    model.fit(train_matrix_selected, train_label_selected)

    test_matrix_selected = train_matrix[split_number:]
    test_label_selected = train_labels[split_number:]

    result = model.predict(test_matrix_selected)
    # print confusion_matrix(train_labels,result)

    print accuracy_score(test_label_selected, result)

def accuracy_ten_fold(model, train_matrix, train_labels):
    print ("Doing 10 fold validation : ")
    print ""
    accuracy = 0
    start = 0
    number_of_test = 10 * len(train_matrix) / 100

    for i in range(0, 10):
        print("Doing iteration " + str(i+1) + " : ")
        train_matrix_selected = train_matrix[:start] + train_matrix[(start+number_of_test):]
        train_label_selected = train_labels[:start] + train_labels[(start+number_of_test):]

        test_matrix_selected = train_matrix[start:(start+number_of_test)]
        test_label_selected = train_labels[start:(start+number_of_test)]

        model.fit(train_matrix_selected, train_label_selected)

        result = model.predict(test_matrix_selected)
        # print confusion_matrix(train_labels,result)

        curr_acc = accuracy_score(test_label_selected, result)
        accuracy = accuracy + curr_acc
        start = start + number_of_test
        print("Accuracy : " + str(curr_acc))

    print accuracy / 10

if __name__ == '__main__':
    (word_dict_frek, word_dict_2, train_labels) = build_word_dictionary()
    train_matrix = convert_to_matrix(word_dict_frek, word_dict_2)

    model1 = MultinomialNB()
    model2 = LinearSVC()
    model3 = GaussianNB()
    model4 = SVC()

    print ""
    print ("Naive Bayes (Gaussian) Learning : ")
    if(len(sys.argv) == 2):
        accuracy_split(model3, train_matrix, train_labels, int(sys.argv[1]))
    else:
        accuracy_ten_fold(model3, train_matrix, train_labels)

    print ""
    print ("Naive Bayes (Multinomial) Learning : ")
    if(len(sys.argv) == 2):
        accuracy_split(model1, train_matrix, train_labels, int(sys.argv[1]))
    else:
        accuracy_ten_fold(model1, train_matrix, train_labels)

    print ""
    print ("SVM (SVC) Learning : ")
    if(len(sys.argv) == 2):
        accuracy_split(model4, train_matrix, train_labels, int(sys.argv[1]))
    else:
        accuracy_ten_fold(model4, train_matrix, train_labels)

    print ""
    print ("SVM (LinearSVC) Learning : ")
    if(len(sys.argv) == 2):
        accuracy_split(model2, train_matrix, train_labels, int(sys.argv[1]))
    else:
        accuracy_ten_fold(model2, train_matrix, train_labels)
