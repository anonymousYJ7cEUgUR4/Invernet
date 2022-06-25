import os
import joblib

from Data.constants import getCurrentArtifactPath

global target_word

def setTargetWord(_target_word):
    global target_word
    target_word = _target_word

def store_metaData(modelToOutput, outputToModel):
    modelToOutputPath = os.path.join(getCurrentArtifactPath(), 'meta/{}_modelToOutput'.format(target_word))
    joblib.dump(modelToOutput, modelToOutputPath)
    
    outputToModelPath = os.path.join(getCurrentArtifactPath(), 'meta/{}_outputToModel'.format(target_word))
    joblib.dump(outputToModel, outputToModelPath)

def store_inferenceModel(inferenceModel):
    inferenceModel.save_weights(os.path.join(getCurrentArtifactPath(), 'model/{}.h5'.format(target_word)))

def store_testEmb(emb):
    embPath = os.path.join(getCurrentArtifactPath(), 'test/{}_emb'.format(target_word))
    joblib.dump(emb, embPath)

def store_testData(testData):
    testDataPath = os.path.join(getCurrentArtifactPath(), 'test/{}_testData'.format(target_word))
    joblib.dump(testData, testDataPath)

def store_yTrue(yTrue):
    yTruePath = os.path.join(getCurrentArtifactPath(), 'test/{}_yTrue'.format(target_word))
    joblib.dump(yTrue, yTruePath)

def store_yPred(yPred):
    yPredPath = os.path.join(getCurrentArtifactPath(), 'output/{}_yPred'.format(target_word))
    joblib.dump(yPred, yPredPath)

def store_yScores(yScores):
    yScoresPath = os.path.join(getCurrentArtifactPath(), 'output/{}_yScores'.format(target_word))
    joblib.dump(yScores, yScoresPath)

