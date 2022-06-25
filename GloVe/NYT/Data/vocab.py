import numpy as np

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

def buildVocabSet(data):
    vocab_set = set()

    for item in data:
        # print(item)
        for word in item:
            vocab_set.add(word)

    return vocab_set

def countVocabFreq(data, vocab_set):
    vocab_freq = {}

    # Initializing a dictionary where <key, value> == <word, term-freq>
    for word in vocab_set:
        vocab_freq[word] = 0
    
    # Counting term-frequency from data
    for item in data:
        for word in item:
            vocab_freq[word] += 1

    # Sorting the vocab according to their term-frequency in a descending order
    sorted_vocab_freq = sorted(vocab_freq, key=vocab_freq.get, reverse=True)
    
    return vocab_freq

def buildTargetWordNeighbors(data, target_word, n_grams=3):
    target_word_neighbors = set()
    
    for item in data:
        sent = item
        indices = np.where(np.array(sent)==target_word)[0]
            
        for idx in indices:
            for i in range(idx-1, idx-n_grams-1, -1):
                if(i >= 0):
                    target_word_neighbors.add(sent[i])
            
            for i in range(idx+1, idx+n_grams+1, 1):
                if(i < len(sent)):
                    target_word_neighbors.add(sent[i])

    return target_word_neighbors







