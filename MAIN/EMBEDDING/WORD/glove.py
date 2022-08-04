import numpy as np
import pandas as pd


def glove_train(input_data, language):
    from scipy import spatial
    from tqdm import tqdm
    from MAIN.CLEANING.basic_cleaning import finalpreprocess

    emmbed_dict = {}
    with open('D:/NLP/NLP_basic//glove/glove.6B.50d.txt','r',encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:],'float32')
            emmbed_dict[word]=vector
    f.close()

    input_data_new = pd.DataFrame()
    # input_data=input_data.iloc[:100]
    columns_list = list(input_data.columns)

    print("Cleaning Started")
    for col in columns_list:
        cleaned_column = f"{col}"
        input_data_new[cleaned_column] = input_data[col].progress_apply(lambda x: finalpreprocess(x,language))
    print("Cleaning Completed")
    
    print("Convert all the reviews in vector form below")
    X = pd.DataFrame()
    for col in input_data_new.columns:
        for doc in tqdm(input_data_new[col].values):
            X[col]=document_vector(doc, emmbed_dict)

    print("GLOVE word embedding completed")

    print(X)
    return X

def document_vector(doc, emmbed_dict):
    # remove out-of-vocabulary words
    doc = [word for word in doc.split() if word in emmbed_dict.keys()]
    sum=0
    for value in doc:
        sum=sum+emmbed_dict[value]
    return sum/len(doc)

