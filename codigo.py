#%%writefile streamlit.py

import numpy as np
import pandas as pd
import streamlit as st
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle   ## For model loading 
import spacy  ## For NLP tasks 
import time

from PIL import Image   ## For image
from io import StringIO  ## for text input and output from the web app

st.title("EDA FAKE NEWS")
st.header("We can see an interactive App about our Fake News EDA built with streamlit")
st.header("")

#EDA
my_dataset = 'train_df.csv'

#Load Dataset
@st.cache(persist = True)
def explore_data(dataset):
  df = pd.read_csv(os.path.join(dataset))
  return df

data = explore_data(my_dataset)

if st.checkbox("Show Dataset"):
  if st.button("Head"):
    st.write(data.head())
  elif st.button("Tail"):
    st.write(data.tail())
  else: 
    st.write(data.head(2))  

st.text("")

#Show entire dataset
if st.checkbox("Show All Dataset"):
  st.dataframe(data) 

st.text("")

#Show Column Name
if st.checkbox("Show Column Names"):
  st.write(data.columns) 

st.text("")

#Show dimensions
data_dimensions = st.radio("What Dimensions Do You Want to See", ("Rows", "Columns", "All"))
if data_dimensions == "Rows":
  st.text("Showing Rows")
  st.write(data.shape[0])
elif data_dimensions == "Columns":
  st.text("Showing Columns")
  st.write(data.shape[1])
else:
  st.text("Showing Shape of Dataset")
  st.write(data.shape)

st.text("")

#Show SetRevisionSummary
if st.checkbox("Show Summary of Dataset"):
  st.write(data.describe())

st.text("")

#Select a Column
col_option = st.selectbox("Select Column", ("Body ID","Stance", "pos_tags_body", "pos_tags_headline", "wordnet_pos_body", 
"wordnet_pos_headline", "lemmatized_body_string", "lemmatized_headline_string"))
if col_option == "Body ID":
  st.write(data['Body ID'])
elif col_option == "Stance":
  st.write(data['Stance'])
elif col_option == "pos_tags_body":
  st.write(data['pos_tags_body'])
elif col_option == "pos_tags_headline":
  st.write(data['pos_tags_headline'])
elif col_option == "wordnet_pos_body":
  st.write(data['wordnet_pos_body'])
elif col_option == "wordnet_pos_headline":
  st.write(data['wordnet_pos_headline'])
elif col_option == "lemmatized_body_string":
  st.write(data['lemmatized_body_string'])
elif col_option == "lemmatized_headline_string":
  st.write(data['lemmatized_headline_string'])
else:
  st.write("Select Column")

st.text("")

#Plot

def plot_CountArticlesByStance():

  stances = data['Stance']

  pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]
  fmt = '{x:,.0f}'
  plt.figure(figsize=(12.8,6))
  ax = stances.value_counts().plot(kind='bar', color=pkmn_type_colors,rot=0)
  ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(fmt))
  for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
  plt.xlabel("Stance", labelpad=14)
  plt.ylabel("Articles", labelpad=14)
  plt.title("Count Articles by Stance", y=1.02);

def plot_PercentatgeArticlesByStance():
  pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

  train_df_by_stances = data.groupby('Stance')['Body ID'].count()
  TotalArticles = train_df_by_stances[0:].sum()
  train_df_by_stances['Percent of Total'] = train_df_by_stances[0:]*100 / TotalArticles
  print(train_df_by_stances['Percent of Total'])

  plt.figure(figsize=(18.6,6))
  ax= train_df_by_stances['Percent of Total'].plot(kind='bar', color=pkmn_type_colors)
  labels = list(train_df_by_stances.index)
  ax.set_xlabel(labels)
  ax.yaxis.set_major_formatter(mtick.PercentFormatter())

  for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points')

  plt.xlabel("Stance", labelpad=14)
  plt.ylabel("% of Articles", labelpad=14)
  plt.title("% of Articles in each Stance", y=1.02);


plot_option = st.selectbox("Some graphics about our Dataset", ("Count Articles By Stance", "Percentatge of Articles By Stance"))

if plot_option == "Count Articles By Stance":
  st.text("0 = Fake, 1 = Fact")
  st.write(plot_CountArticlesByStance())
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.pyplot()
if plot_option == "Percentatge of Articles By Stance":
  st.text("0 = Fake, 1 = Fact")
  st.write(plot_PercentatgeArticlesByStance())
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.pyplot()



########################################################################################################################################


st.title("Predicting Fake News")
st.header('This App is created to predict if a New is real or fake')

image = Image.open('fakeNews.jpeg')

st.image(image, caption='Fake News')

headline = st.text_area('Enter a headline')
outputHeadline = ""

headline = st.text_area('Enter the body')
outputBody = ""

if st.button("Predict"):
  st.success(f"The news item is {outputHeadline}")
  st.balloons()
  left_column, right_column = st.beta_columns(2)