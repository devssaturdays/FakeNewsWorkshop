#%%writefile streamlit.py
import streamlit.components.v1 as components
import streamlit as st

import numpy as np
import pandas as pd
import os
import pickle   ## For model loading 
import time
import nltk
import string
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split


#from PIL import Image   ## For image
from io import StringIO  ## for text input and output from the web app
########################################################################################################################################

def Load_model():
  with open('model_gradientBoostingMachine', 'rb') as f:
    model = pickle.load(f)
  return model

def Load_Tfidf():
  with open('tfidf', 'rb') as f:
    tfidf = pickle.load(f)
  return tfidf  

def Load_PCA():
  with open('pca', 'rb') as f:
    pca = pickle.load(f)
  return pca  

def get_wordnet_pos(tag):
  """
  Function to convert tags into wordnet's format
  """
  if tag.startswith('J'):
      return wordnet.ADJ
  elif tag.startswith('V'):
      return wordnet.VERB
  elif tag.startswith('N'):
      return wordnet.NOUN
  elif tag.startswith('R'):
      return wordnet.ADV
  else:
      return wordnet.NOUN

def list_to_string(list_):

  '''
  This function takes the list and put all the elements of the list to a string with 
  space as a separator
  '''
  final_string = str()
  for element in list_:
      final_string += str(element) + " "
  return final_string

def clean_data(body, headline):
  df = pd.DataFrame({'Body':body, 'Headline':headline}, index=[0])

  #Text pre processing
  df['Body'] = df['Body'].str.lower().replace("\r", " ").replace("\n", " ").replace("    ", " ").replace('"', '').replace('[^\w\s]','').replace("'s", "")
  df['Headline'] = df['Headline'].str.lower().replace("\r", " ").replace("\n", " ").replace("    ", " ").replace('"', '').replace('[^\w\s]','').replace("'s", "")

  #Tokenization
  df['tokenized_body'] = df['Body'].apply(word_tokenize)
  df['tokenized_headline'] = df['Headline'].apply(word_tokenize)

  #Removing stopwords
  stop_words = set(stopwords.words('english'))
  punctuation = string.punctuation  + "”" + "“" + "’" + "``" + "‘"


  df['stopwords_removed_body'] = df['tokenized_body'].apply(lambda x: [word for word in x if word not in stop_words])
  unwanted_characters=["",""]
  df['stopwords_removed_body']  = df['stopwords_removed_body'].apply(lambda x: [word.replace("'", "") for word in x])
  df['stopwords_removed_body']  = df['stopwords_removed_body'].apply(lambda x: [word for word in x if word not in punctuation])


  df['stopwords_removed_headline'] = df['tokenized_headline'].apply(lambda x: [word for word in x if word not in stop_words])
  df['stopwords_removed_headline']  = df['stopwords_removed_headline'].apply(lambda x: [word.replace("'", "") for word in x])
  df['stopwords_removed_headline']  = df['stopwords_removed_headline'].apply(lambda x: [word for word in x if word not in punctuation])

  #Lemmatization
  df['pos_tags_body'] = df['stopwords_removed_body'].apply(nltk.tag.pos_tag)
  df['pos_tags_headline'] = df['stopwords_removed_headline'].apply(nltk.tag.pos_tag)

  df['wordnet_pos_body'] = df['pos_tags_body'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
  df['wordnet_pos_headline'] = df['pos_tags_headline'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

  wnl = WordNetLemmatizer()

  df['lemmatized_body'] = df['wordnet_pos_body'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
  df['lemmatized_headline'] = df['wordnet_pos_headline'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

  df['lemmatized_body_string'] = df.lemmatized_body.apply(list_to_string)
  df['lemmatized_headline_string'] = df.lemmatized_headline.apply(list_to_string)


  df['lemmatized_body_string'] = df['lemmatized_body_string'].str.lower().replace("\r", " ").replace("\n", " ").replace("    ", " ").replace('"', '').replace('[^\w\s]','').replace("'s", "")
  df['lemmatized_headline_string'] = df['lemmatized_headline_string'].str.lower().replace("\r", " ").replace("\n", " ").replace("    ", " ").replace('"', '').replace('[^\w\s]','').replace("'s", "")

  df =  df[['lemmatized_body_string','lemmatized_headline_string']]


  #TF-IDF
  tfidf = Load_Tfidf()

  tf_idf_headlines = tfidf.fit_transform(df['lemmatized_headline_string']).toarray() 
  tf_idf_bodies = tfidf.fit_transform(df['lemmatized_body_string']).toarray()

  features = np.concatenate((tf_idf_bodies,tf_idf_headlines),axis=1)

  #PCA
  pca = Load_PCA()
  # #Apply pca to features
  features_reduced = pca.fit_transform(features) 

  return features_reduced

def runPrediction():

  st.subheader("Detect Fake News with our App")

  st.write('Introduce only a headline and the body to detect if it is fake or not.')

  headline = st.text_area("Enter a headline here", height=200)

  body = st.text_area("Enter the body here", height = 500)

  

  if st.button("Predict"):

    new_data = clean_data(body, headline)

    st.write(new_data)

    model = Load_model()

    #stance = model.predict(new_data)

    st.subheader('New Stance')
    #st.subheader('The new is' + str(np.stance[0]))
    st.balloons()

def introduction():
  st.title("Fake News")
  st.header('Disinformation in the age of the information society')

  #image = Image.open('Images/fakeNews.jpeg')

  st.image(image, width=650)

  st.write('There have always been fake news, but with the emergence of the Internet and new communication and information technologies, fake news has become part of our daily lives. Social media algorithms do not have an easy task determining the truthfulness of information. In this way, supposedly real images and videos that have been professionally manipulated are disseminated. This disinformation can influence debates and public opinion and cause a hugh damage on our society and daily lifes.')
  st.text('\n')
  st.write('Unfortunately, very few people are able to detect fake news or dont have time to verify if the information they are reading is real or false. This should not be a problem for people, because thanks to artificial intelligence we will be able to detect whether it is a fake or a real news.')

