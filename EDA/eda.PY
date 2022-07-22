
def pie_chart(text,numbers):
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Pie(labels=text, values=numbers,hole=.5,textinfo='label+percent+value',title='Similarity graph'
    )])
    fig.update_layout(title_text="Similarity graph",title_x=0.5,template="plotly_dark")

    return fig

def charts(data,input_column,output_column,chart,language):
    labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']  ## for testing purpose
    values = [4500, 2500, 1053, 500] ## for testing purpose
    if chart == "similarity-graph":
        text, numbers = fetch_labels_values(data,input_column,output_column,language)
        fig = pie_chart(text, numbers)
        return fig
    else: 
        pass

def fetch_labels_values(data,input_column,output_column,language):

    ##imports
    from CLEANING.basic_cleaning import finalpreprocess
    from EMBEDDING.tfidf import tfidf_Vec_eda,tfidf_Vec_train
    from tqdm import tqdm
    tqdm.pandas()

    cleaned_column = f"{input_column}_cleaned"
    vector_column = f"{cleaned_column}_vectors"
    data[cleaned_column] = data[input_column].progress_apply(lambda x: finalpreprocess(x,language))
    
    tfidf_vectors = tfidf_Vec_eda(data[cleaned_column])
    inter_dict = similarity_graph(tfidf_vectors)
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


def get_similarity_Score(inter_dict):

    count = 0
    for i in inter_dict.values():
        count+=i

    similarity_score = (count/len(inter_dict))*100
    values = [similarity_score,100-similarity_score]
    labels = ["Similar","Non Similar"]
    return values,labels