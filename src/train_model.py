from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

def train_word2vec_model(text):
    sentences = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")