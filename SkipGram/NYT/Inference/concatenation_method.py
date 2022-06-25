import warnings
from matplotlib.pyplot import get
import numpy as np
from random import randint
import joblib

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Attention, Concatenate, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback, LearningRateScheduler

from Data.constants import getPath, getCurrentArtifactPath
from Embedding.w2v import getPretrainedModel, getPretrainedEmb, getVocabSize, getModelIdxToOutputIdx
from Inference.input_output import getInputSrc, getInputDown, getOutput
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------

global inferenceModel

class ProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        line = '\rEpoch: {}, Loss: {:e}'.format(epoch, logs.get('loss'))
        print(line, end='')
        
    def on_train_end(self, logs=None):
        print('')

class EarlyStopping(Callback):
    def __init__(self, monitor='loss', value=-1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
            
        if current < self.value:
            print("Epoch %05d: early stopping THR" % epoch)
            
            self.model.stop_training = True

def lr_sched(epoch, lr):
    if epoch < 150:
        return 0.1
    elif epoch < 300:
        return 0.01
    elif epoch < 500:
        return 0.001
    elif epoch < 1000:
        return 0.0001

def buildInferenceModel():
    global inferenceModel

    src_input = Input(shape=(20), batch_size=getVocabSize())
    down_input = Input(shape=(20), batch_size=getVocabSize())

    src_attention = Attention()([src_input, src_input])
    down_attention = Attention()([down_input, down_input])

    input_layer = Concatenate()([src_input, down_input])

    dense1 = Dense(512, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal')(input_layer)
    drop1 = Dropout(0.3)(dense1)
    dense2 = Dense(128, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal')(drop1)
    drop2 = Dropout(0.3)(dense2)
    dense3 = Dense(32, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal')(drop2)
    dense4 = Dense(1, name='output', activation='sigmoid')(dense3)

    inferenceModel = Model(inputs=[src_input, down_input], outputs=dense4)

    inferenceModel.compile(loss=CosineSimilarity(axis=0), optimizer=SGD(learning_rate=0.01, clipnorm=1))

def trainInferenceModel(_epochs=500, _early_loss=-1):
    global inferenceModel

    inferenceModel.fit([getInputSrc(), getInputDown()], getOutput(), verbose=0, epochs=_epochs, batch_size=getVocabSize(), callbacks=[ProgressCallback(), LearningRateScheduler(lr_sched), EarlyStopping(value=_early_loss)])

    return inferenceModel

def generateClassificationScores(yTrue, yPred):
    precision, recall, f1, support = precision_recall_fscore_support(yTrue, yPred, average='macro', labels=np.unique(yPred))

    # print('Precision: {}'.format(precision)) # Reducing false positives
    # print('Recall: {}'.format(recall)) # Reducing false negatives
    # print('F1-Score: {}'.format(f1)) # For imbalanced classification

    return precision, recall, f1

def refineSigmoidOutput(yPred, threshold=0.5):
    return [1 if i > threshold else 0 for i in yPred.T.squeeze()]

def trainingEvaluation(numInferenceSamples):
    global inferenceModel

    sampleNo = randint(0, numInferenceSamples-1)

    emb = joblib.load(getPath('emb_{}'.format(sampleNo)))
    yTrue = joblib.load(getPath('cooc_binary_{}.txt'.format(sampleNo)))

    p = 'null'
    r = 'null'
    f1 = 'null'

    try:
        prediction = inferenceModel.predict([getPretrainedEmb(), emb], batch_size=getVocabSize())

        # For sigmoid getOutput()
        yPred = refineSigmoidOutput(prediction, threshold=0.5)

        p, r, f1 = generateClassificationScores(yTrue.squeeze(), yPred)
    except:
        print("In training evaluation, an exception has occurred during prediction")

    return p, r, f1

def testingEvaluation(w2vModel, target_word, target_word_neighbors):
    global inferenceModel
    
    emb = joblib.load(getPath('emb_test'))
    yTrue = joblib.load(getPath('cooc_binary_test.txt'))

    maxP, maxR, tn, fp, fn, tp = 0, 0, 0, 0, 0, 0
    maxF1, maxAuc = -1, -1

    try:
        prediction = inferenceModel.predict([getPretrainedEmb(), emb], batch_size=getVocabSize())

        thresholds = [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.01, 0.001, 0.0001]
        maxF1 = 0
        for _threshold in thresholds:
            yPred = refineSigmoidOutput(prediction, threshold=_threshold)

            # Discarding words that are not present in the downstream model - start
            presentIdx = []
            for item in getPretrainedModel().key_to_index:
                if item in target_word_neighbors and item in w2vModel.wv.key_to_index:
                    presentIdx.append(getModelIdxToOutputIdx()[getPretrainedModel().key_to_index[item]])

            vec1 = []
            vec2 = []
            vec3 = []

            for i in range(len(yTrue.T)):
                if i in presentIdx:
                    vec1.append(yTrue.squeeze()[i])
                    vec2.append(yPred[i])
                    vec3.append(prediction.T.squeeze()[i])

            yTrue = np.array(vec1)
            yPred = np.array(vec2)
            yScores = np.array(vec3)
            # Discarding words that are not present in the downstream model - end

            p, r, f1 = generateClassificationScores(yTrue, yPred)

            if f1 > maxF1:
                maxP, maxR, maxF1 = p, r, f1 
                tn, fp, fn, tp = confusion_matrix(yTrue, yPred).ravel()
                maxAuc = roc_auc_score(yTrue, yScores, max_fpr=1, labels=[0, 1])

        # print('Max F1: {}'.format(maxF1))
    except:
        print("In testing evaluation, an exception has occurred during prediction")

    return yTrue, yPred, yScores, maxP, maxR, maxF1, tn, fp, fn, tp, maxAuc