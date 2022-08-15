import numpy as np
import pandas as pd

def document_vector(model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc.split() if word in model.wv.index_to_key]
    return np.mean(model.wv[doc], axis=0)

def word2vec_model(story):
    import gensim
    model = gensim.models.Word2Vec(window=9, min_count=2)
    model.build_vocab(story)
    model.train(story, total_examples=model.corpus_count, epochs=100)
    return model

def word2vec_train(input_data, language):
    
    from gensim.utils import simple_preprocess
    from tqdm import tqdm
    from MAIN.CLEANING.basic_cleaning import finalpreprocess
    from tqdm import tqdm
    tqdm.pandas()

    X = []
    story = []
    input_data_new = pd.DataFrame()
    # input_data=input_data.iloc[:10]
    columns_list = list(input_data.columns)

    print("Cleaning Started")
    for col in columns_list:
        cleaned_column = f"{col}"
        input_data_new[cleaned_column] = input_data[col].progress_apply(lambda x: finalpreprocess(x,language))
    print("Cleaning Completed")
    
    #convert sentences to word tokens 
    for col in input_data_new.columns:
        for doc in input_data_new[col]:
            story.append(simple_preprocess(doc))
    
    print("word2vec model training started")
    model= word2vec_model(story)
    print("word2vec model training completed")

    #convert all the column values into 100 dimension vector
    print("Convert all the reviews in vector form below")
    X = pd.DataFrame()
    for col in input_data_new.columns:
        for doc in tqdm(input_data_new[col].values):
            X[col]=document_vector(model , doc)

    print("WORD2VEC word embedding completed")
    return model, X

def word2vec_pretrained_model(input_data, language):
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    from tqdm import tqdm
    from MAIN.CLEANING.basic_cleaning import finalpreprocess
    from tqdm import tqdm
    tqdm.pandas()

    X = []
    story = []
    input_data_new = pd.DataFrame()
    input_data=input_data.iloc[:10]
    columns_list = list(input_data.columns)

    print("Cleaning Started")
    for col in columns_list:
        cleaned_column = f"{col}"
        input_data_new[cleaned_column] = input_data[col].progress_apply(lambda x: finalpreprocess(x,language))
    print("Cleaning Completed")
    
    #convert sentences to word tokens 
    for col in input_data_new.columns:
        for doc in input_data_new[col]:
            story.append(simple_preprocess(doc))
    
    print("word2vec model training started")
    model = Word2Vec(story, min_count=2)
    print("word2vec model training completed")

    #convert all the column values into 100 dimension vector
    print("Convert all the reviews in vector form below")
    X = pd.DataFrame()
    for col in input_data_new.columns:
        for doc in tqdm(input_data_new[col].values):
            X[col]=document_vector(model , doc)

    print("WORD2VEC word embedding completed")
    return model, X