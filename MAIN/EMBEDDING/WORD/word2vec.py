
# from calendar import month_name, monthrange


# def word2vec_train(data,mode='cbow'):
    
#     import gensim
#     from gensim.models import Word2Vec
#     import nltk

#     tokens = [nltk.word_tokenize(i) for i in data]  

#     if mode == 'cbow':
#         # Create CBOW model
#         model1 = Word2Vec(tokens, min_count = 3,
#                               vector_size = 100, window = 5)
#     else:
#         # Create Skip Gram model
#         model2 = Word2Vec(tokens, min_count = 3, vector_size = 100,
#                                              window = 5, sg = 1)

import gensim
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from tqdm import tqdm
import numpy as np
import pandas as pd

def document_vector(model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc.split() if word in model.wv.index_to_key]
    return np.mean(model.wv[doc], axis=0)


def word2vec_train(input_data=df['review']):
    X = []
    story = []
    
    #convert sentences to word tokens 
    for doc in input_data:
        story.append(simple_preprocess(doc))
    
    # model training
    model = gensim.models.Word2Vec(window=9, min_count=2)
    model.build_vocab(story)
    model.train(story, total_examples=model.corpus_count, epochs=100)
    
    #convert all the column values into 100 dimension vector
    for doc in tqdm(input_data.values):
        X.append(document_vector(model, doc))
    X = np.array(X)
    return X

X = word2vec_train(input_data=df['review'])
    
    
