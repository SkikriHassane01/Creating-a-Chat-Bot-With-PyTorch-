# ===========Import libraries============================
import nltk
# nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer = PorterStemmer()

# =============tokenization function======================
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# ===============steaming function========================
def stem(word):
    return stemmer.stem(word, to_lowercase=True)

# ===============bag of words function====================
def bag_of_words(pattern_sentence, all_words):
    """
    1 for each known word that exists in the sentence, 0 otherwise
    sentence = ["hello", "how", "are", "you"]
    all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    
    pattern_sentence = [stem(word) for word in pattern_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for idx, word in enumerate(all_words):
        if word in pattern_sentence:
            bag[idx] = 1.0
    return bag