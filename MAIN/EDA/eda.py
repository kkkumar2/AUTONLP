
def pie_chart(text,numbers):
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Pie(labels=text, values=numbers,hole=.5,textinfo='label+percent+value',title='Similarity graph'
    )])
    fig.update_layout(title_text="Similarity graph",title_x=0.5,template="plotly_dark")

    return fig

def bar_chart(text,numbers):
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
            x=numbers,
            y=text,
            orientation='h'))
    fig.update_layout(title_text="Word-frequency graph",title_x=0.5,template="seaborn")
    
    return fig

def fetch_labels_values(data,input_column,category_column,language):

    ##imports
    from CLEANING.basic_cleaning import finalpreprocess
    from EMBEDDING.SENTENCE.tfidf import tfidf_Vec_similarity,tfidf_Vec_train
    from EMBEDDING.SENTENCE.bert import bert_similarity
    from tqdm import tqdm
    tqdm.pandas()

    cleaned_column = f"{input_column}_cleaned"
    vector_column = f"{cleaned_column}_vectors"
    data[cleaned_column] = data[input_column].progress_apply(lambda x: finalpreprocess(x,language))
    # data[vector_column] = tfidf_Vec_similarity(data[cleaned_column])
    data[vector_column] = bert_similarity(data[cleaned_column].to_list())

    if category_column:
        cat = data[category_column].value_counts().index.to_list()
        cat_str = data[category_column].value_counts().index.to_list()
        for i in range(len(cat)):
            cat[i] = data[data['Service'] == cat[i]]
        text = []; numbers = []
        for i in range(len(cat)):
            cat[i] = cat[i].reset_index(drop=True)
            if len(cat[i]) > 10:
                inter_dict = similarity_graph(cat[i][vector_column])
                values,labels = get_similarity_Score(inter_dict,cat_str[i])
                text.extend(labels)
                numbers.extend(values)
            else:
                continue
        return text,numbers
        
    else:
        # tfidf_vectors = tfidf_Vec_eda(data[cleaned_column])
        # inter_dict = similarity_graph(tfidf_vectors)
        inter_dict = similarity_graph(data[vector_column])
        values,labels = get_similarity_Score(inter_dict)

        return labels,values


def cosine_similarity(v1, v2):

    ##imports
    import numpy as np

    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)

def similarity_graph(vectors):

    ##imports
    from tqdm import tqdm, tqdm_pandas
    #tqdm_pandas(tqdm())
    tqdm.pandas()

    inter_dict = {}
    for i in tqdm(range(len(vectors))):
        mid_score = []
        for j in range(len(vectors)):
            if i == j:
                continue
            else:
                # score = cosine_similarity(vectors.numpy()[i], vectors.numpy()[j])
                score = cosine_similarity(vectors[i], vectors[j])
                mid_score.append(score)
        mid_sort_score = mid_score.copy()
        mid_sort_score.sort(reverse=True)
        index1 = mid_score.index(mid_sort_score[0])
        index2 = mid_score.index(mid_sort_score[1])
        index3 = mid_score.index(mid_sort_score[2])
        avg_score = (mid_sort_score[0] + mid_sort_score[1] + mid_sort_score[2])/3
        inter_dict[f"{i},{(index1,index2,index3)},{(mid_sort_score[0],mid_sort_score[1],mid_sort_score[2])}"] = avg_score
    print("Sucessfully Completed")
    return inter_dict


def get_similarity_Score(inter_dict,cat=""):

    count = 0
    for i in inter_dict.values():
        count+=i

    similarity_score = (count/len(inter_dict))*100
    values = [similarity_score,100-similarity_score]
    labels = [f"{cat}_Similar",f"{cat}_Non Similar"]
    return values,labels


def fetch_word_frequency(content):
    from collections import Counter
    cnt = Counter()
    for text in content:
        for ele1 in text.split(' '):
            if len(ele1) > 3:
                cnt[ele1] += 1
    texts = []
    numbers = []
    for ele in cnt.most_common(50):
        texts.append(ele[0])
        numbers.append(ele[1])
    return texts,numbers


def generate_wordcloud(text,title = None): 

    import matplotlib.pyplot as plt
    from wordcloud import WordCloud,STOPWORDS
    from nltk.corpus import stopwords
    stop_word = set(STOPWORDS)
    
    #Creating wordcloud object and getting list of frequent words from text
    wordcloud = WordCloud(background_color='black',
                          max_words=len(text),
                          max_font_size=40, 
                          scale=3,
                          stopwords = stop_word,).generate(text)
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    return fig

def plot_ngrams(text, n=2, topk=50,language="english"):

    print("ngrams")
    import matplotlib.pyplot as plt
    from nltk.util import ngrams
    from collections import Counter
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words(language))
  
    word_tokens = word_tokenize(text)
    tokens = [w for w in word_tokens if not w.lower() in stop_words]
    # get the ngrams 
    ngram_phrases = ngrams(tokens, n)
    
    # Get the most common ones 
    most_common = Counter(ngram_phrases).most_common(topk)
    
    # Make word and count lists 
    words, counts = [], []
    for phrase, count in most_common:
        word = ' '.join(phrase)
        words.append(word)
        counts.append(count)
    
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.bar(words,counts)
    plt.xlabel("n-grams found in the text")
    plt.ylabel("Ngram frequencies")
    plt.xticks(rotation=90)
    return fig