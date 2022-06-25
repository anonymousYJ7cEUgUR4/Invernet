import os

global currentArtifactPath

def getPath(fileName):
    return os.path.join('/home/<user_name>/Documents/Experiments/Scripts/SkipGram_1/AG_NEWS/cooc_ensemble', fileName)

def setArtifactPath(n, b):
    global currentArtifactPath

    currentArtifactPath = os.path.join('/home/<user_name>/Documents/Experiments/Scripts/SkipGram_1/AG_NEWS/artifacts', 
                                       'SkipGram_n{}_b{}'.format(n, b))

    return currentArtifactPath

def getCurrentArtifactPath():
    global currentArtifactPath
    
    return currentArtifactPath