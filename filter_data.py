import sys
import json
import csv

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

#----------------------------------------------------------------------
def csv_writer(data, path):
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)
#----------------------------------------------------------------------
if __name__ == "__main__":
    data = []
    path = "output.csv"
    with open('spam2.csv', 'rb') as spamcsv:
        readCSV = csv.reader(spamcsv)
        for row in readCSV:
            if is_ascii(row[1]):
                data.append(row)
#             sentence = row[1]
#             spam = row[0] # spam = 0, not spam = 1
    csv_writer(data, path)
