# Extract sentences from books.

import sys
import codecs
import string
import re
import os
from spacy.en import English


def extract(data_path):
    """
    Extract sentences from books.
    :param data_path: path to the dir with books.
    """
    # The first allowable character in a sentence
    header_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z', '"', "'"]
    nlp = English()
    # List names of the files
    list_dir = os.listdir(data_path)
    print('There are %s files for training.' % len(list_dir))
    sent_num = 0
    # Create the output file with sentences.
    output_file = codecs.open('sentences.txt', encoding='utf-8', mode='w')
    for file in list_dir:
        # Make path for every file.
        file_path = '/'.join([data_path, file])
        input_file = codecs.open(file_path, encoding='utf-8', mode='r', errors='ignore')
        text = []
        for line in input_file:
            text.append(line.strip())
        text = ' '.join(text)
        # Make Spacy doc-object
        doc = nlp(text)
        sentences = []
        for sent in doc.sents:
            sentences.append(str(sent))
        for sent in sentences:
            sent = sent.strip()
            sent = sent.replace("\t", " ")
            sent = re.sub(' +', ' ', sent)
            # Take only sentences with length more then 10 words with first symbol
            # in the header_list and second symbol - not in the header list.
            if (len(sent) > 10) and (sent[0] in header_list) and (sent[1] not in header_list):
                output_file.write(sent + '\n')
                sent_num += 1
                sys.stdout.write("\rSentence number %s." % sent_num)
                sys.stdout.flush()

if __name__ == '__main__':
    path = str(sys.argv[1])
    extract(path)
