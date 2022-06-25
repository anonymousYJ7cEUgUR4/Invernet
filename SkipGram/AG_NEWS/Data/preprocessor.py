import re

import nltk
from nltk.corpus import stopwords
stop_words = set(nltk.corpus.stopwords.words('english'))

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

def clean(doc):
    # replacing non-word/whitespace characters by a single space
    new_doc = re.sub(r'[^\w\s]', ' ', doc)
    # replacing all digits/numbers by a special tag '<num>'
    new_doc = re.sub(r'[0-9]+', '<num>', new_doc)
    # replacing all new line characters with a single space
    new_doc = re.sub(r'\n', ' ', new_doc)
    
    # converting to lowercase
    new_doc = new_doc.lower()
    
    return new_doc

def removeStopWords(docs):
    new_docs_list = []

    #Removing stop words
    for sentence in docs:
        split_ = sentence.split()
        split_ = [word for word in split_ if word not in stop_words]
        
        new_docs_list.append(split_)

    return new_docs_list