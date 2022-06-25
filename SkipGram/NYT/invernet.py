import numpy as np
from contextlib import redirect_stdout
import gc

import tensorflow as tf
import tensorflow.python.keras.backend as K

from Data.constants import setArtifactPath
from Data.loader import loadData, splitDataset, trimDataset
from Data.vocab import buildVocabSet, countVocabFreq, buildTargetWordNeighbors
from Data.storage import *
from Embedding.w2v import getPretrainedModel, pretrain, indexPretrainedEmbedding
from Inference.input_output import prepareAndStoreTrainingIO, readTrainingIO, prepareAndStoreTestingIO
from Inference.concatenation_method import buildInferenceModel, testingEvaluation, trainInferenceModel, trainingEvaluation
from Log.csv_writer import setFileName, getFileName, writeHeader, writeRow

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

def setupGPU():
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 1})
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)

def setupResultLog(fileName):
    setFileName(fileName)
    writeHeader(['target_word', 'training_p', 'training_r', 'training_f1', 'testing_p', 'testing_r', 'testing_f1', 'testing_tn', 'testing_fp', 'testing_fn', 'testing_tp', 'testing_auc'])

if __name__ == '__main__':
    # n_arr = [5, 15, 30]
    n_arr = [5, 15, 30]
    b_arr = [5, 25, 50]

    for n in n_arr:
        for b in b_arr:
            hyperparameters = {
                    'pretrain_epochs': 0,
                    'inference_epochs': 0,

                    'num_inference_samples': n,
                    'bin_size': b
                }

            setupGPU()
            print('GPU setup complete')

            with open('./logs/training_log_SkipGram_n{}_b{}.txt'.format(hyperparameters['num_inference_samples'], hyperparameters['bin_size']), 'w') as logFile:
                # with redirect_stdout(logFile):
                    setArtifactPath(n, b)

                    setupResultLog('./results/results_SkipGram_n{}_b{}.csv'.format(hyperparameters['num_inference_samples'], hyperparameters['bin_size']))
                    print('Result log setup complete')

                    print('\nLoading data ...')
                    data, labelsDictionary = loadData()
                    print('Data loaded')

                    vocabSet = list(buildVocabSet(data))
                    vocabFreq = countVocabFreq(data, vocabSet)

                    target_words = [k for k in vocabFreq if vocabFreq[k] >= 50]
                    vocabIndices = np.random.randint(0, len(target_words), size=(1, round(0.1*len(target_words)))).squeeze()
                    target_words = [target_words[i] for i in vocabIndices]

                    # target_words = ['small']
                
                    for target_word in target_words:
                        print('\nTarget word: {}'.format(target_word))

                        setTargetWord(target_word)

                        if vocabFreq[target_word] > 10000:
                            hyperparameters['pretrain_epochs'] = 50
                            hyperparameters['inference_epochs'] = 1000
                        elif vocabFreq[target_word] > 1000:
                            hyperparameters['pretrain_epochs'] = 100
                            hyperparameters['inference_epochs'] = 800
                        elif vocabFreq[target_word] >= 50:
                            hyperparameters['pretrain_epochs'] = 1000
                            hyperparameters['inference_epochs'] = 500

                        print('Trimming dataset ...')
                        dataTrimmed = trimDataset(data, target_word)

                        print('Building target word neighbors ...')
                        target_word_neighbors = buildTargetWordNeighbors(dataTrimmed, target_word)

                        training_p, training_r, training_f1 = 'null', 'null', 'null'
                        maxP, maxR, maxF1, maxTn, maxFp, maxFn, maxTp, maxAuc = 0, 0, 0, 0, 0, 0, 0, 0
                        for i in range(0, 3):
                            print('Splitting dataset ...')
                            sourceData, inferenceData, downstreamData = splitDataset(dataTrimmed, labelsDictionary)

                            if len(sourceData) == 0 or len(inferenceData) == 0 or len(downstreamData) == 0:
                                continue

                            print('Pretraining embedding ...')
                            pretrainedModel, vocabSize = pretrain(_epochs=hyperparameters['pretrain_epochs'])

                            print('Indexing embedding ...')
                            pretrainedEmb, output_idx_to_model_idx, model_idx_to_output_idx = indexPretrainedEmbedding(target_word, target_word_neighbors)

                            print('Preparing and storing training IO ...')
                            prepareAndStoreTrainingIO(hyperparameters['num_inference_samples'], hyperparameters['bin_size'], target_word, target_word_neighbors)

                            print('Reading training IO ...')
                            readTrainingIO(hyperparameters['num_inference_samples'])

                            print('Building inference model ...')
                            buildInferenceModel()

                            print('Training inference model ...')
                            inferenceModel = trainInferenceModel(_epochs=hyperparameters['inference_epochs'])

                            print('Evaluating on one random inference sample ...')
                            training_p, training_r, training_f1 = trainingEvaluation(hyperparameters['num_inference_samples'])

                            print('Preparing and storing testing IO ...')
                            testModel, testData = prepareAndStoreTestingIO(target_word, target_word_neighbors)
                        
                            print('Evaluating on downstream data')
                            yTrue, yPred, yScores, testing_p, testing_r, testing_f1, tn, fp, fn, tp, auc = testingEvaluation(testModel, target_word, target_word_neighbors)                           

                            if testing_f1 > maxF1:
                                maxP, maxR, maxF1, maxTn, maxFp, maxFn, maxTp, maxAuc = testing_p, testing_r, testing_f1, tn, fp, fn, tp, auc

                                store_metaData(model_idx_to_output_idx, output_idx_to_model_idx)
                                store_inferenceModel(inferenceModel)
                                store_testData(testData)
                                store_yTrue(yTrue)
                                store_yPred(yPred)
                                store_yScores(yScores) 

                        print('Target word: {}, F1: {}'.format(target_word, maxF1))

                        if maxF1 != -1 and maxAuc != -1:
                            writeRow([target_word, training_p, training_r, training_f1, maxP, maxR, maxF1, maxTn, maxFp, maxFn, maxTp, maxAuc])

                        gc.collect()








