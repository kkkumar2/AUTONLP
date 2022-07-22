import streamlit as st
import os
import pandas as pd
from EDA import eda
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
            input_column = st.multiselect("Select Input column",cols)
            output_column = st.multiselect("Select output column",cols)
            usecase = st.selectbox("Select output column",config.USECASES)
    else:
        st.error("File is improper")


elif app_mode == "Train":
    pass

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
            output_column = st.multiselect("Select output column",cols)
            language = st.multiselect("Select Language",config.LANGUAGE)
            chart = st.multiselect("Select EDA Graph",config.CHART_AVAIL)
            # if input_column is not None & output_column is not None & chart is not None:
            fetch = st.button("Create chart")
            if len(input_column) > 0 and len(output_column) > 0 and len(chart) > 0:
                if fetch:
                    fig = eda.charts(df,input_column[0],output_column[0],chart[0],language[0])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("File is improper")