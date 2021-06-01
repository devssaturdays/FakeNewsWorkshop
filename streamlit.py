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

def getDataInfo():
  st.write("The first step in the Machine Learning process is getting data.")
  st.write("This process depends on your project and data type. For example, are you planning to collect real-time data from an IoT system or static data from an existing database?")
  st.write("You can also use data from internet repositories sites such as Kaggle and others.")
  st.write("In our case training and testing examples were taken from http://www.fakenewschallenge.org/")

def prepareDataInfo():

  col1, col2, col3 = st.beta_columns([1,6,1])

  st.write("Real-world data often has unorganized, missing, or noisy elements. Therefore, for Machine Learning success, after we chose our data, we need to clean, prepare, and manipulate the data.")
  st.write("This process is a critical step, and people typically spend up to 80% of their time in this stage. Having a clean data set helps with your model’s accuracy down the road.")
  st.write("Our data is in text format so we had to perform a text preprocessing. What is text prepocessing? To preprocess your text simply means to bring your text into a form that is predictable and analyzable for your task. ")
  st.write("There are different ways to preprocess your text. The techniques that we have used for preprocessing are the following:")

  st.write("1. Tokenization: is the text preprocessing task of breaking up text into smaller components of text (known as tokens).")
  #image = Image.open('Images/tokenize.png')
  st.image(image, width=600)

  st.write("2. Removing stop words: since there's a lot of prepositions (in, of, to) and conjunctions (and, but, or, nor, for, so, yet) and ponctuation (, - .) and definite or indefinite articles (a,an,the) in our data, it might be useful to introduce a key concept on Natural Language Processing stop words. Stop words refers to the most common words in a language and when dealing with text processing they shoul be removed, since they do not add any valuable information to our studies.")
  #image = Image.open('Images/stopWords.png')
  st.image(image, width=600)
  
  st.write("3. Lemmatization: is a linguistic term that means grouping together words with the same root or lemma but with different inflections or derivatives of meaning so they can be analyzed as one item. The aim is to take away inflectional suffixes and prefixes to bring out the word’s dictionary form.")
  #image = Image.open('Images/lemmatization.png')
  st.image(image, width=300)
  
  st.write("4. TF-IDF: is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.")
  #image = Image.open('Images/tf-idf.png')
  st.image(image, width=600)
  
  st.write("For the use of all these techniques we have used NLTK. Is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written. Here you can find more information: https://www.nltk.org/.")

def choose_modelInfo():
  st.write("There are different models that we can choose according to the objective we have: we will use algorithms for classification , prediction , linear regression ,  clustering (example k-means or k-nearest neighbor ),   Deep Learning  (ex:  neural network ), Bayesian , etc. There are variants if what we are going to process are images, sound, text, numerical values.") 
  st.write("In our case we have chosen Gradient Boosting. Gradient Boosting  is a family of algorithms used in both classification and regression based on the combination of weak predictive models ( weak learners ) -usually decision trees- to create a predictive model. strong. Weak decision trees are generated sequentially, with each tree being created to correct the errors in the previous tree. The apprentices are typically shallow trees , typically just one, two, or three levels deep.")

  #image = Image.open('Images/gradientBoosting.png')

  st.image(image, width=600)

  st.write("For the implementation of the model we have used Scikit-learn. Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms. Here you can find more information: https://scikit-learn.org/stable/.")



  
