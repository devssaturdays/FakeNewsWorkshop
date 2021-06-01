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

st.write("hello")
  
