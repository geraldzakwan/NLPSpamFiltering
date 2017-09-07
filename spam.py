import sys
import json
import csv
import nltk
from nltk.stem import WordNetLemmatizer

# reload(sys)
# sys.setdefaultencoding("utf-8")
# nltk.download()
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

    with open('spam.csv', 'rb') as spamcsv:
        readCSV = csv.reader(spamcsv)
        for row in readCSV:
            sentence = row[1]
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

    return word_dictionary

if __name__ == '__main__':
    print(build_word_dictionary())
