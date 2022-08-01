def charts(data,input_column,category_column,chart,language):

    if chart == "similarity-graph":
        from MAIN.EDA.eda import fetch_labels_values
        from MAIN.EDA.eda import pie_chart

        text, numbers = fetch_labels_values(data,input_column,category_column,language)
        return pie_chart(text, numbers)
    elif chart == "word-frequency":
        from MAIN.EDA.eda import fetch_word_frequency
        from MAIN.EDA.eda import bar_chart

        texts,numbers = fetch_word_frequency(data[input_column].values)
        return bar_chart(texts, numbers) 
    elif chart == "stopwords": 
        pass
    
    from MAIN.EDA.eda import plot_ngrams

    if chart == "bi-gram": 
        return plot_ngrams(" ".join(data[input_column].values), n=2, topk=50,language=language)
    elif chart == "tri-gram": 
        return plot_ngrams(" ".join(data[input_column].values), n=3, topk=50,language=language)
    elif chart == "four-gram": 
        return plot_ngrams(" ".join(data[input_column].values), n=4, topk=50,language=language)
    elif chart == "word-cloud":
        from MAIN.EDA.eda import generate_wordcloud
        return generate_wordcloud(" ".join(data[input_column].values))


def classification(classification_model,word_embedding,df,labels):
    import pandas as pd
    if word_embedding == "WORD2VEC":
        from MAIN.EMBEDDING.WORD.word2vec import word2vec_train
        print("Inside Classificataion")
        X= word2vec_train(df, "english")
        print(type(X))
        print(len(X))
        
# classification("abc","WORD2VEC","abc","labels")


def ner(df,sentence,word,pos,label,ner_model):
    if ner_model == "BILSTM":
        from MAIN.MODELS.NER import bilstm
        out = bilstm.preprocess(df,sentence,word,pos,label)
        return out
    else:
        pass

    return "success"