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


def classification(classification_model,word_embedding,feature,labels):
    pass