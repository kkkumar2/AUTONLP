
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

def word2vec_train(input_data, language):
    input_data=input_data.iloc[:10]
    columns_list = list(input_data.columns)
    print("Inside Word2Vec")
    from MAIN.CLEANING.basic_cleaning import finalpreprocess
    from tqdm import tqdm
    tqdm.pandas()
    
    print("Cleaning Started")
    for col in columns_list:
        cleaned_column = f"{col}_cleaned"
        input_data[cleaned_column] = input_data[col].progress_apply(lambda x: finalpreprocess(x,language))
    print(input_data.columns)
    print("Cleaning Completed")
    X = []
    story = []
    #convert sentences to word tokens 
    for doc in input_data[cleaned_column]:
        story.append(simple_preprocess(doc))
    
    # model training
    model = gensim.models.Word2Vec(window=9, min_count=2)
    model.build_vocab(story)
    print("word2vec model training started")
    model.train(story, total_examples=model.corpus_count, epochs=100)
    print("word2vec model training completed")
    
    print(len(model.wv.index_to_key))
    #convert all the column values into 100 dimension vector
    print("Convert all the reviews in vector form")
    for doc in tqdm(input_data[cleaned_column].values):
        X.append(document_vector(model, doc))
    # X = np.array(X)
    return X

# def word2vec_train(input_data, input_column, language):
#     print("Inside Word2Vec")
#     from CLEANING.basic_cleaning import finalpreprocess
#     from tqdm import tqdm
#     tqdm.pandas()
#     cleaned_column = f"{input_column}_cleaned"
#     # vector_column = f"{cleaned_column}_vectors"
#     print("Cleaning Started")
#     input_data[cleaned_column] = input_data[input_column].progress_apply(lambda x: finalpreprocess(x,language))
#     print("Cleaning Completed")
#     X = []
#     story = []
#     #convert sentences to word tokens 
#     for doc in input_data[cleaned_column]:
#         story.append(simple_preprocess(doc))
    
#     # model training
#     model = gensim.models.Word2Vec(window=9, min_count=2)
#     model.build_vocab(story)
#     print("word2vec model training started")
#     model.train(story, total_examples=model.corpus_count, epochs=100)
#     print("word2vec model training completed")
    
#     print(len(model.wv.index_to_key))
#     #convert all the column values into 100 dimension vector
#     print("Convert all the reviews in vector form")
#     for doc in tqdm(input_data[cleaned_column].values):
#         X.append(document_vector(model, doc))
#     # X = np.array(X)
#     return X
    
    
