# ===========Import libraries============================
import nltk
# nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# =============tokenization function======================
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# ===============steaming function========================
def stem(word):
    return stemmer.stem(word, to_lowercase=True)

# ===============bag of words function====================
def bag_of_words(tokenized_sentence, all_words):
    pass

# testing 
string = "How long this project will tack"
print (f"string: {string}")
print (f"tokenized string: {tokenize(string)}")

words = ['organize', "organizes", "organizing"]
print (f"words: {words}")
stemmed_words = [stem(w) for w in words]
print (f"steaming result: {stemmed_words}")