from datasets import load_dataset

from Data.preprocessor import clean, removeStopWords

import json
import random
import math

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

global sourceData, inferenceData, downstreamData

def loadData():
    dataset = load_dataset("ag_news")  
    data = dataset['train']['text'][:] 
    labels = dataset['train']['label'][:] 

    cleaned_data = [clean(doc) for doc in data]
    data = removeStopWords(cleaned_data)

    labelsDictionary = {}
    for i in range(len(data)):
        labelsDictionary[' '.join(data[i])] = labels[i]

    return data, labelsDictionary

# Trim the dataset so that only the sentences containing the target_word are included
def trimDataset(data, target_word):
    trimmed_dataset = [sent for sent in data if target_word in sent]

    return trimmed_dataset

# Split the dataset randomly into source (50%), inference (40%), and downstream (10%) datasets
def splitDataset(data, labelsDictionary):
    global sourceData, inferenceData, downstreamData

    sec = random.randint(0, 3)

    # Downstream data is coming from only section sec
    downstreamArticles = [data[i] for i in range(len(data)) if labelsDictionary[' '.join(data[i])]==sec]
    nonDownstreamArticles = [data[i] for i in range(len(data)) if labelsDictionary[' '.join(data[i])]!=sec]

    random.shuffle(downstreamArticles)
    
    nonDownstreamArticles = nonDownstreamArticles + downstreamArticles[100:len(downstreamArticles)]
    downstreamArticles = downstreamArticles[0:100]
    
    source_count = math.floor(len(nonDownstreamArticles) * 0.50)
    inference_count = math.floor(len(nonDownstreamArticles) * 0.50)

    random.shuffle(nonDownstreamArticles)

    sourceData = nonDownstreamArticles[0:source_count]
    inferenceData = nonDownstreamArticles[source_count:source_count + inference_count]
    downstreamData = downstreamArticles

    return sourceData, inferenceData, downstreamData

#------------------------GETTERS------------------------#

def getSourceData():
    global sourceData
    return sourceData

def getInferenceData():
    global inferenceData
    return inferenceData

def getDownstreamData():
    global downstreamData
    return downstreamData