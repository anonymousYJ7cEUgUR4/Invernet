import gc
from os import path, system
import numpy as np
import joblib
import csv
from mittens.np_mittens import Mittens

from Data.constants import getPath

#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------

global pretrainedModel, pretrainedKeysList, pretrainedEmb, vocabSize, model_idx_to_output_idx, output_idx_to_model_idx

def glove2Dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:]))) for line in reader}
        
        return embed

def pretrain(_epochs=100, _vector_size=20, _min_count=3, _window_size=10):
    global pretrainedModel, pretrainedKeysList, vocabSize

    pretrainedModel = None

    # vector_size is 20 to prevent overfitting
    vectorSize = str(_vector_size)
    epochs = str(_epochs)
    minCount = str(_min_count)
    windowSize = str(_window_size)
    
    system('./glove_pretrain.sh {} {} {} {} {} {} {} {} {}'.format('./cooc_ensemble/source.txt', './cooc_ensemble/source_vocab', './cooc_ensemble/source_cooc', './cooc_ensemble/source_cooc_shuf', './cooc_ensemble/pretrained', minCount, vectorSize, epochs, windowSize))

    pretrainedModel = glove2Dict('./cooc_ensemble/pretrained.txt')
    pretrainedKeysList = list(pretrainedModel)
    vocabSize = len(pretrainedModel)

    return pretrainedModel, vocabSize

def indexPretrainedEmbedding(target_word, target_word_neighbors):
    global pretrainedModel, pretrainedKeysList, pretrainedEmb, vocabSize, model_idx_to_output_idx, output_idx_to_model_idx

    pretrainedEmb = np.zeros((vocabSize, 20), dtype=np.float32)

    output_idx_to_model_idx = {}
    model_idx_to_output_idx = {}

    idx = 0
    for word in pretrainedKeysList:
        if word in target_word_neighbors or word==target_word:            
            pretrainedEmb[idx] = pretrainedModel[word]

            output_idx_to_model_idx[idx] = pretrainedKeysList.index(word)
            model_idx_to_output_idx[pretrainedKeysList.index(word)] = idx

            idx += 1
        
    return pretrainedEmb, output_idx_to_model_idx, model_idx_to_output_idx

def fineTune(data, count, _vector_size=20, _max_iter=1000):
    global pretrainedKeysList

    #Fine-tuning GloVe Model

    #---------------------------------------------------------------------------------------------------------

    # Building vocab for fine tuning pretrained GloVe embeddings using Mittens
    vocab = set()
    for sent in data:
        for tok in sent:
            vocab.add(tok)
    vocab = list(vocab)
    
    vocabSize = len(vocab)
    
    vocabIdx = {}
    for i, word in enumerate(vocab):
        vocabIdx[word] = i
    
    #---------------------------------------------------------------------------------------------------------
    
    # Building cooccurrence matrix for fine tuning pretrained GloVe embeddings using Mittens
    n_grams = 3
    cooc_mat = np.zeros((vocabSize, vocabSize))
    
    for i, word in enumerate(vocab):
        for sent in data:
            sent = np.array(sent)        
            indices = np.where(sent==word)[0]

            for idx in indices:
                for j in range(idx-1, idx-n_grams-1, -1):
                    if((j >= 0) and (sent[j] in pretrainedKeysList)):
                        cooc_mat[i][vocabIdx[sent[j]]] += 1
                        cooc_mat[vocabIdx[sent[j]]][i] += 1

                for j in range(idx+1, idx+n_grams+1, 1):
                    if((j < len(sent)) and (sent[j] in pretrainedKeysList)):
                        cooc_mat[i][vocabIdx[sent[j]]] += 1
                        cooc_mat[vocabIdx[sent[j]]][i] += 1
    
    #---------------------------------------------------------------------------------------------------------
    
    mittensModel = Mittens(n=_vector_size, max_iter=_max_iter)
    newEmbeddings = mittensModel.fit(cooc_mat, vocab=vocab, initial_embedding_dict=pretrainedModel)
    
    print('GloVe Model {} Training Complete\n'.format(count))
    
    gloveModel = {}
    
    for i, word in enumerate(vocab):
        gloveModel[word] = newEmbeddings[i]

    del vocab, cooc_mat
    gc.collect()
    
    return gloveModel

def indexAndStoreDownstreamEmbedding(downstreamModel, target_word, target_word_neighbors, i):
    global pretrainedModel, pretrainedKeysList, pretrainedEmb, model_idx_to_output_idx
    emb = pretrainedEmb.copy()
    
    downstreamKeysList = list(downstreamModel)

    idx = model_idx_to_output_idx[pretrainedKeysList.index(target_word)]
    emb[idx] = downstreamModel[target_word]
        
    for word in downstreamKeysList:
        if word in target_word_neighbors and word in pretrainedKeysList:
            idx = model_idx_to_output_idx[pretrainedKeysList.index(word)]

            emb[idx] = np.array(downstreamModel[word], copy=True)

    emb_path = getPath('emb_{}'.format(i))
    joblib.dump(emb, emb_path)

def buildCoocVec(sentences, target_word, n_gram_window=1):     
    global pretrainedModel, pretrainedKeysList, vocabSize, model_idx_to_output_idx

    pretrainedKeysList = list(pretrainedModel)

    cooc_vec_actual = np.zeros((vocabSize))
    cooc_vec_binary = np.zeros((vocabSize))
    
    for sent in sentences:
        sent = np.array(sent)
        indices = np.where(sent==target_word)[0]
        
        for idx in indices:
            for i in range(idx-1, idx-n_gram_window-1, -1):
                if((i >= 0) and (sent[i] in pretrainedKeysList)):
                    cooc_vec_actual[model_idx_to_output_idx[pretrainedKeysList.index(sent[i])]] += 1
            
            for i in range(idx+1, idx+n_gram_window+1, 1):
                if((i < len(sent)) and (sent[i] in pretrainedKeysList)):
                    cooc_vec_actual[model_idx_to_output_idx[pretrainedKeysList.index(sent[i])]] += 1    
    
    cooc_vec_binary[cooc_vec_actual>0] = 1
    
    return cooc_vec_binary.reshape(1, vocabSize)

#------------------------GETTERS------------------------#

def getPretrainedModel():
    global pretrainedModel
    return pretrainedModel

def getPretrainedEmb():
    global pretrainedEmb
    return pretrainedEmb

def getVocabSize():
    global vocabSize
    return vocabSize

def getModelIdxToOutputIdx():
    global model_idx_to_output_idx
    return model_idx_to_output_idx

def getOutputIdxToModelIdx():
    global output_idx_to_model_idx
    return output_idx_to_model_idx
