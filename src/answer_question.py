from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

def load_model():
    model = Word2Vec.load("word2vec.model")
    return model

def get_answer(model, question, text):
    sentences = sent_tokenize(text)
    question_tokens = word_tokenize(question.lower())
    most_similar_sentence = None
    max_similarity = -1
    
    for sentence in sentences:
        sentence_tokens = word_tokenize(sentence.lower())
        similarity = model.wv.n_similarity(question_tokens, sentence_tokens)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_sentence = sentence
    
    return most_similar_sentence