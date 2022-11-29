#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: hamzafarooq@ MABA CLASS
"""

import pandas as pd
import re
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm




st.title("Dubai Hotel Finder")
st.subheader("Find a hotel in Dubai via review similarity")
query = st.text_input("Start typing")
# =============================================================================
# st.markdown("This is a demo Streamlit app.")
# st.markdown("My name is Hamza, hello world!..")
# st.markdown("This is v2")
# =============================================================================

#@st.cache(persist=True)

url = 'https://raw.githubusercontent.com/nicsusuki/first-streamlit-app/main/hotelReviewsInDubai__en2019100120191005.csv'
df = pd.read_csv(url)

df['hotelName'] = df.hotelName.str.split('\n').str[0]
df['hotelName'] = df.hotelName.str.split('   ').str[1]
df['hotelName'].drop_duplicates()
df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(''.join).reset_index(name='all_review')
  
df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))
df = df_combined

df_sentences = df.set_index("all_review")
df_sentences = df_sentences["hotelName"].to_dict()
df_sentences_list = list(df_sentences.keys())

df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# Corpus with example sentences
corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)



# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))

query_embedding = embedder.encode(query, convert_to_tensor=True)

# We use cosine-similarity and torch.topk to find the highest 5 scores
cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
top_results = torch.topk(cos_scores, k=top_k)

    #print("\n\n======================\n\n")
st.success("Query:", query)
    #print("\nTop 5 most similar sentences in corpus:")

for score, idx in zip(top_results[0], top_results[1]):
    st.success("(Score: {:.4f})".format(score))
    st.success(corpus[idx], "(Score: {:.4f})".format(score))
    row_dict = df.loc[df['all_review']== corpus[idx]]
    st.success("paper_id:  " , row_dict['hotelName'] , "\n")
  
    
    
    
    #disp_head = st.sidebar.radio('Select DataFrame Display Option:',('Head', 'All'),index=0)



    #Multi-Select
    #sel_plot_cols = st.sidebar.multiselect("Select Columns For Scatter Plot",df.columns.to_list()[0:4],df.columns.to_list()[0:2])

    #Select Box
    #x_plot = st.sidebar.selectbox("Select X-axis Column For Scatter Plot",df.columns.to_list()[0:4],index=0)
    #y_plot = st.sidebar.selectbox("Select Y-axis Column For Scatter Plot",df.columns.to_list()[0:4],index=1)


# =============================================================================
#     if disp_head=="Head":
#         st.dataframe(df.head())
#     else:
#         st.dataframe(df)
# =============================================================================
    #st.table(df)
    #st.write(df)


    #Scatter Plot
# =============================================================================
#     fig = px.scatter(df, x=df["sepallength"], y=df["sepalwidth"], color="class",
#                  size='petallength', hover_data=['petalwidth'])
# 
#     fig.update_layout({
#                 'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
# 
#     fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
# 
#     st.write("\n")
#     st.subheader("Scatter Plot")
#     st.plotly_chart(fig, use_container_width=True)
# 
# =============================================================================

    #Add images
    #images = ["<image_url>"]
    #st.image(images, width=600,use_container_width=True, caption=["Iris Flower"])




# =============================================================================
# 
# if __name__ == '__main__':
#     run()
# =============================================================================
