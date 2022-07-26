
def tfidf_Vec_train(data):

    ##imports
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
    import os

    tfidf_vectorizer = TfidfVectorizer(use_idf=True,max_df=0.5)
    vectors = tfidf_vectorizer.fit_transform(data) 
    with open(os.path.join("Embedding","similarity.features"),"wb") as f:
        joblib.dump(vectors,f)
    

    
def tfidf_Vec_similarity(data):

    ##imports
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(use_idf=True,max_df=0.5)
    vectors = tfidf_vectorizer.fit_transform(data)
    vectors = vectors.toarray()
    vec_list = []
    for ele in vectors:
        vec_list.append(ele)
    return vec_list