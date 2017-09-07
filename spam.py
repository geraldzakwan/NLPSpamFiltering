import json,csv
import nltk
from nltk.stem import WordNetLemmatizer

# nltk.download()
if __name__ == '__main__':
    # get json file stop words
    file = open('stopwords-en.json','r+')
    stopwords_list = []
    stopwords_list = json.load(file)
    # print(stopwords_list)
    file.close()
    lemmatizer = WordNetLemmatizer()
    print(lemmatizer.lemmatize('monsters'))
    # with open('spam.csv') as spamcsv:
        #readCSV = csv.reader(csvfile,delimiter=',,,')
        # for row in readCSV:
            # lemmatization
            # filterword = lemmatizer.lemmatize()

            # hapus kata2 dari filterword jika ada di list stop words

            # hapus tanda baca , ! & ? 
            
            # make dict