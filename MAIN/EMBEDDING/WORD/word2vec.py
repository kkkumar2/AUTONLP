
from calendar import month_name, monthrange


def word2vec_train(data,mode='cbow'):
    
    import gensim
    from gensim.models import Word2Vec
    import nltk

    tokens = [nltk.word_tokenize(i) for i in data]  

    if mode == 'cbow':
        # Create CBOW model
        model1 = Word2Vec(tokens, min_count = 3,
                              vector_size = 100, window = 5)
    else:
        # Create Skip Gram model
        model2 = Word2Vec(tokens, min_count = 3, vector_size = 100,
                                             window = 5, sg = 1)
