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
import time
import pickle as pkl
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.title("Dubai Hotel Finder")

#Add images
images = ["https://whatson.ae/wp-content/uploads/2017/08/downtown-facebook.jpg"]
st.image(images, width=600,use_column_width=True, caption=["Dubai Skyline"])
st.markdown("This hotel finder looks at Trip Advisor reviews of hotels in Dubai and returns hotels with reviews similar to the search text.")
st.text('Data was obtained from:')
st.text('https://www.kaggle.com/datasets/hamzafarooq50/hotel-listings-and-reviews?select=hotelReviewsInDubai__en2019100120191005.csv')
st.subheader("Find a hotel in Dubai via review similarity")
query = st.text_input("Type search terms here")

#@st.cache(persist=True)

url = 'https://raw.githubusercontent.com/nicsusuki/first-streamlit-app/main/hotelReviewsInDubai__en2019100120191005.csv'
df = pd.read_csv(url)

df['hotelName'] = df.hotelName.str.split('\n').str[0]
df['hotelName'] = df.hotelName.str.split('   ').str[1]
df['hotelName'].drop_duplicates()
df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(' '.join).reset_index(name='all_review')
  
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
# corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)

# open the corpus_embeddings file
fileo = open('corpus_embeddings.pkl' , "rb")
# loading data
corpus_embeddings = pkl.load(fileo)

# open the summary file
fileo = open('review_summary.pkl' , "rb")
# loading data
summary = pkl.load(fileo)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 1

query_embedding = embedder.encode(query, convert_to_tensor=True)

# We use cosine-similarity and torch.topk to find the highest 5 scores
cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
top_results = torch.topk(cos_scores, k=top_k)

search_button = st.button('Search')

st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )

if search_button:
    st.markdown("**Finding hotel best matching:** " + query)
    with st.spinner("Generating results..."):
        time.sleep(2)
        #st.markdown('***Top 5 most similar hotels based on reviews:***')
        if not isinstance(query, str) or not len(query) > 1:
                         st.markdown("No search terms found.")
        else:
            for score, idx in zip(top_results[0], top_results[1]):
                n = int(top_results[1])
                row_dict = df.loc[df['all_review']== corpus[idx]]
                s = pd.Series(row_dict['hotelName'])
                st.success(s.to_string(index=False) + "  (Score: {:.4f})".format(score))
                st.subheader('Summary of all Reviews for {}:' .format(s.to_string(index=False)))
         
                st.markdown(summary.iloc[n][0])
                
              

            st.set_option('deprecation.showPyplotGlobalUse', False)  
            #get index for top review
            n = int(top_results[1])
            # Create some sample text
            text = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x)).apply(lambda x: lower_case(x))[n]
            # Create and generate a word cloud image:
            wordcloud = WordCloud().generate(str(text))
            
            # Display the generated image:
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot()    
    
    
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

    




# =============================================================================
# 
# if __name__ == '__main__':
#     run()
# =============================================================================
