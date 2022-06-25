from datasets import load_dataset

from Data.preprocessor import clean, removeStopWords

import json
import random
import math

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

global sourceData, inferenceData, downstreamData

def loadData():
    dataset_file = open('../../NYTDataset/articles-search-1990-2016.json')
    dataset = json.load(dataset_file)

    sectionDict = {'World': 0, 'Arts': 1, 'Sports': 2, 'Opinion': 3}
    labelsDictionary = {}

    dataRaw = [(str(item['lead']), sectionDict[item['section']])  for item in dataset if item['section']=='World' or item['section']=='Arts' or item['section']=='Sports' or item['section']=='Opinion']
    
    data = []
    for i in range(len(dataRaw)):
        cleaned_data = clean(dataRaw[i][0])
        nostop_data = removeStopWords(cleaned_data)
        
        data.append(nostop_data)
        labelsDictionary[' '.join(nostop_data)] = dataRaw[i][1]

    return data, labelsDictionary

# Trim the dataset so that only the sentences containing the target_word are included
def trimDataset(data, target_word):
    trimmed_dataset = [item for item in data if target_word in item]

    return trimmed_dataset

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

def writeSourceDataset(data):
    # Dataset for GloVe
    with open('./cooc_ensemble/source.txt', 'w') as file:
        for sent in data:
            sentence = ' '.join([tok for tok in sent])

            file.write(sentence)
            file.write('\n')

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