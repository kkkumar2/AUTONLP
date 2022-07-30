import streamlit as st
import os
import pandas as pd
from MAIN import run
import config

def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads',exist_ok=True)
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1,os.path.join('uploads',uploaded_file.name)
    except Exception as e:
        print(F"Error is {e}")
        return 0


app_mode = st.sidebar.selectbox('Application mode',
['About App','Upload','Train','Test','EDA'])
if app_mode =='About App':
    with open("README.md", "r", encoding="utf-8") as fh:
        readme = ""
        unwanted_list = ['<h2>','![GIF]','## Dataset','<a href=','A demo']
        for line in fh:            
            if line.startswith(tuple(unwanted_list)): 
                continue
            readme = readme + line
    st.markdown(readme)

elif app_mode == "Upload":
    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file is not None:
        status,path = save_uploaded_file(uploaded_file)
        if status:
            df = pd.read_csv(path)
            print(df.columns)
            cols = df.columns
            # features = st.multiselect("Select Input column",cols)
            # labels = st.multiselect("Select output column",cols)
            # if len(features) > 0 and len(labels) > 0:
            usecase = st.selectbox("Select Usecase",config.USECASES)
            if usecase == "classification":
                features = st.multiselect("Select Input column",cols)
                labels = st.multiselect("Select output column",cols)
                if len(features) > 0 and len(labels) > 0:
                    classification_model = st.selectbox("Select CLASSIFICATION MODELS",config.CLASSIFICATION_MODELS)
                    word_embedding = st.selectbox("Select WORD EMDEDDING ",config.WORD_EMDEDDING)
                    train = st.button("Start training")
                    if train:
                        from MAIN.run import classification

                        out = classification(classification_model,word_embedding,df[features],df[labels])
            if usecase == "similarity":
                pass
            if usecase == "NER":
                sentence = st.selectbox("Select sentence column",cols)
                word = st.selectbox("Select text column",cols)
                pos = st.selectbox("Select pos column",cols)
                label = st.selectbox("Select label column",cols)
                if sentence is not None and word is not None and pos is not None and label is not None:
                    ner_model = st.selectbox("Select NER model",config.NER_MODELS)
                    if ner_model:
                        train = st.button("Start training")
                        if train:
                            from MAIN.run import ner

                            out = ner(df,sentence,word,pos,label,ner_model)
                            st.success(f"{ner_model} model trained with accuracy {out}")
    # else:
    #     st.error("File is improper")


# elif app_mode == "Train":
#     pass

elif app_mode == "Test":
    pass

elif app_mode == "EDA":
    import plotly.graph_objects as go

    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file is not None:
        status,path = save_uploaded_file(uploaded_file)
        if status:
            df = pd.read_excel(path)
            cols = df.columns
            input_column = st.multiselect("Select Input column",cols)
            language = st.multiselect("Select Language",config.LANGUAGE)
            chart = st.multiselect("Select EDA Graph",config.CHART_AVAIL)
            if len(chart):
                if chart[0] == "similarity-graph":
                    category_column = st.multiselect("Select any one category column",cols)
            # else:
            #     category_column = [None]
            fetch = st.button("Create chart")
            if len(input_column) > 0 and len(language) > 0 and len(chart) > 0:
                if fetch:
                    try:
                        if len(category_column) == 0:
                            category_column = [None]
                    except Exception as e:
                        category_column = [None]
                    fig = run.charts(df,input_column[0],category_column[0],chart[0],language[0])
                    if fig:
                        if chart[0] in ['word-cloud','word-cloud','bi-gram','tri-gram','four-gram']:
                            st.pyplot(fig)
                        else:
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("File is improper")