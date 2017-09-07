import json,csv
import nltk
from nltk.stem import WordNetLemmatizer

def make_dictionary(train_dir):
    # train_dir as array of words all training data


# nltk.download()
if __name__ == '__main__':
    # get json file stop words
    file = open('stopwords-en.json','r+')
    stopwords_list = []
    stopwords_list = json.load(file)
    punctuantion_list = ['!','&','?',]
    # print(stopwords_list)
    file.close()
    filter_words = []
    lemmatizer = WordNetLemmatizer()
    # print(lemmatizer.lemmatize('monsters'))
    with open('spam.csv') as spamcsv:
        readCSV = csv.reader(spamcsv)
        for row in readCSV:
            # lemmatization
            # filterword = lemmatizer.lemmatize(row[1])
            # print(row[0])
            words = row[1].split()
            # print(words)
            # hapus kata2 dari filterword jika ada di list stop words            
            for word in words:
                if not word in stopwords_list and not word in punctuantion_list:
                    word_lmt = lemmatizer.lemmatize(word)
                    filter_words.append(word_lmt)
            print(filter_words)