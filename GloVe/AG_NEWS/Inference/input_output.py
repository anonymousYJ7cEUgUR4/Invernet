import os
from numpy.random import default_rng as rng
import numpy as np
import joblib

from Data.constants import getPath
from Data.loader import getInferenceData, getDownstreamData
from Embedding.glove import getPretrainedEmb, getVocabSize, fineTune, indexAndStoreDownstreamEmbedding, buildCoocVec

#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

inputSrc = None
inputDown = None
output = None

def prepareAndStoreTrainingIO(numInferenceSamples, binSize, target_word, target_word_neighbors):
    for i in range(0, numInferenceSamples):        
        data = rng().choice(getInferenceData(), binSize)

        embeddingModel = fineTune(data, i)
        indexAndStoreDownstreamEmbedding(embeddingModel, target_word, target_word_neighbors, i)

        cooc_binary_path = getPath('cooc_binary_{}'.format(i))
        cooc_binary = buildCoocVec(data, target_word, n_gram_window=1)
        joblib.dump(cooc_binary, cooc_binary_path)
        
    del embeddingModel, cooc_binary

def readTrainingIO(numInferenceSamples):
    global inputSrc, inputDown, output

    inputSrc = None
    inputDown = None
    output = None

    for i in range(0, numInferenceSamples):
        embPath = getPath('emb_{}'.format(i))
        emb = joblib.load(embPath)
        
        outputFilePath = getPath('cooc_binary_{}'.format(i))
        outputFile = joblib.load(outputFilePath)
        
        if inputSrc is None:
            inputSrc = getPretrainedEmb().copy()
        else:    
            inputSrc = np.concatenate((getPretrainedEmb(), inputSrc), axis=0)
            
        if inputDown is None:
            inputDown = emb.copy()
        else:
            inputDown = np.concatenate((emb, inputDown), axis=0)
            
        if output is None:
            output = outputFile.copy()
        else:
            output = np.concatenate((outputFile, output), axis=0)
            
    output = np.reshape(output, (numInferenceSamples*getVocabSize(), 1))

def prepareAndStoreTestingIO(target_word, target_word_neighbors):
    embeddingModel = fineTune(getDownstreamData(), 101)
    indexAndStoreDownstreamEmbedding(embeddingModel, target_word, target_word_neighbors, 101)

    #Computing cooccurrence
    cooc_binary_test_path = getPath('cooc_binary_101')
    cooc_binary_test = buildCoocVec(getDownstreamData(), target_word, n_gram_window=1)
    joblib.dump(cooc_binary_test, cooc_binary_test_path)

    return embeddingModel, getDownstreamData()

#------------------------GETTERS------------------------#

def getInputSrc():
    global inputSrc
    return inputSrc

def getInputDown():
    global inputDown
    return inputDown

def getOutput():
    global output
    return output