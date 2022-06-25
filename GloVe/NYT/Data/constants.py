import os

global currentArtifactPath

def getPath(fileName):
    return os.path.join('/home/<user_name>/Documents/Experiments/Scripts/GloVe/cooc_ensemble', fileName)

def setArtifactPath(n, b):
    global currentArtifactPath

    currentArtifactPath = os.path.join('/home/<user_name>/Documents/Experiments/Scripts/GloVe/artifacts', 
                                       'GloVe_n{}_b{}'.format(n, b))

    return currentArtifactPath

def getCurrentArtifactPath():
    global currentArtifactPath
    
    return currentArtifactPath