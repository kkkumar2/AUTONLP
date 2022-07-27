
def tfidf_Vec_eda(data):

    ##imports
    from glove import Corpus, Glove

    corpus = Corpus() 
    corpus.fit(data, window=10)
    glove = Glove(no_components=5, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')

    # vec_list = []
    # for ele in vectors:
    #     vec_list.append(ele)
    # return vec_list