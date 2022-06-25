from os import path
import numpy as np
import joblib
from gensim.models import Word2Vec, KeyedVectors

from Data.loader import getSourceData
from Data.constants import getPath

#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------

global pretrainedModel, pretrainedEmb, vocabSize, model_idx_to_output_idx, output_idx_to_model_idx

def pretrain(_epochs=100, _vector_size=20, _min_count=3):
    global pretrainedModel

    pretrainedModel = None

    # vector_size is 20 to prevent overfitting
    pretrainedModel = Word2Vec(getSourceData(), vector_size=_vector_size, min_count=_min_count, sg=1)
    # pretrainedModel.build_vocab(['UNK'], update=True)

    pretrainedModel.train(getSourceData(), total_examples=len(getSourceData()), epochs=_epochs)

    pretrainedModel.wv.save_word2vec_format('pretrained_w2v.bin', binary=True)
    pretrainedModel = KeyedVectors.load_word2vec_format('pretrained_w2v.bin', binary=True) 

    vocabSize = len(pretrainedModel.key_to_index)

    return pretrainedModel, vocabSize

def indexPretrainedEmbedding(target_word, target_word_neighbors):
    global pretrainedModel, pretrainedEmb, vocabSize, model_idx_to_output_idx, output_idx_to_model_idx

    vocabSize = len(pretrainedModel.key_to_index)
    pretrainedEmb = np.zeros((vocabSize, 20), dtype=np.float32)

    output_idx_to_model_idx = {}
    model_idx_to_output_idx = {}

    idx = 0
    for word in pretrainedModel.key_to_index:
        if word in target_word_neighbors or word==target_word:            
            pretrainedEmb[idx] = pretrainedModel[word]

            output_idx_to_model_idx[idx] = pretrainedModel.key_to_index[word]
            model_idx_to_output_idx[pretrainedModel.key_to_index[word]] = idx

            idx += 1
        
    return pretrainedEmb, output_idx_to_model_idx, model_idx_to_output_idx

def fineTune(data, i, _vector_size=20, _min_count=1, _epochs=100):
    #Fine-tuning word2vec model
    w2vModel = Word2Vec(vector_size=_vector_size, min_count=_min_count, sample=0, sg=1)
    
    w2vModel.build_vocab(data)
    w2vModel.wv.vectors_lockf = np.ones(len(w2vModel.wv))
    w2vModel.wv.intersect_word2vec_format('pretrained_w2v.bin', lockf=1.0, binary=True)
    
    w2vModel.train(data, total_examples=len(data), epochs=_epochs)
    
    # w2vModel.wv.save_word2vec_format(getPath('{}_w2v.bin'.format(i)), binary=True)
    # print('Word2Vec Model {} Training Complete\n'.format(i))
    
    return w2vModel

def indexAndStoreDownstreamEmbedding(downstreamModel, target_word, target_word_neighbors, i):
    global pretrainedModel, pretrainedEmb, model_idx_to_output_idx
    emb = pretrainedEmb.copy()
    
    idx = model_idx_to_output_idx[pretrainedModel.key_to_index[target_word]]
    emb[idx] = downstreamModel.wv[target_word]
        
    for word in downstreamModel.wv.key_to_index:
        if word in target_word_neighbors and word in pretrainedModel.key_to_index:
            idx = model_idx_to_output_idx[pretrainedModel.key_to_index[word]]

            emb[idx] = np.array(downstreamModel.wv[word], copy=True)

    emb_path = getPath('emb_{}'.format(i))
    joblib.dump(emb, emb_path)

def buildCoocVec(sentences, target_word, n_gram_window=1):     
    global pretrainedModel, vocabSize, model_idx_to_output_idx

    cooc_vec_actual = np.zeros((vocabSize))
    cooc_vec_binary = np.zeros((vocabSize))
    
    for sent in sentences:
        sent = np.array(sent)
        
        indices = np.where(sent==target_word)[0]
        
        for idx in indices:
            for i in range(idx-1, idx-n_gram_window-1, -1):
                if((i >= 0) and (sent[i] in pretrainedModel.key_to_index)):
                    cooc_vec_actual[model_idx_to_output_idx[pretrainedModel.key_to_index[sent[i]]]] += 1
            
            for i in range(idx+1, idx+n_gram_window+1, 1):
                if((i < len(sent)) and (sent[i] in pretrainedModel.key_to_index)):
                    cooc_vec_actual[model_idx_to_output_idx[pretrainedModel.key_to_index[sent[i]]]] += 1    
    
    cooc_vec_binary[cooc_vec_actual>0] = 1
    
    cooc_vec_binary_softmax = np.zeros((2, vocabSize), dtype=np.float32)
    cooc_vec_binary_softmax[0][cooc_vec_binary==0] = 1
    cooc_vec_binary_softmax[1][cooc_vec_binary==1] = 1
                        
    return cooc_vec_binary.reshape(1, vocabSize), cooc_vec_binary_softmax.T

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
