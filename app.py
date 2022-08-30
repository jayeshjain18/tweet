import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.porter import PorterStemmer
import nltk
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import string as str
import pickle
model = pickle.load(open('model1.pickle','rb'))
bow_vectorizer= pickle.load(open('vector1.pickle','rb'))
def deletepattern(text,pattern):
    words=re.findall(pattern,text)
    for word in words:
        text=re.sub(word,"",text)
    text=text.__str__()
    text=text.replace("[^a-zA-z#]", " ")
    return text


def remove(s):

    s = s.split()
    x1 = ""

    for w in s:
        if len(w) > 3:
            x1 = x1 + " " + w
            # x="111".join([w])
    tokens =x1.split()
    stemmer = PorterStemmer()

    x1=""

    for y in tokens:
            x1=x1+' '+ stemmer.stem(y)

        #for i in range(len(tokens)):
            #tokens[i] = " ".join(tokens[i])
    x=""
    x=x.join(x1)

    return x

def  predict(x):



    ans=model.predict(bow_vectorizer.transform([x]))
    ans=ans[0]
    return  ans
st.set_page_config(page_title="tweet sentiment analysis by Jayesh",page_icon="🚗 ")
st.title("Tweet's Sentiment Analysis")
text1=st.text_input("please enter your tweet here: ")
if st.button('Predict Tweet'):
    def stop():
           if text1=="":
                st.error("sorry tweet cannot be empty")
                return

           x="@[\w]*"
           s1=deletepattern(text1,x)
           toke=remove(s1)
           ans=predict(toke)

           if ans==0:
               st.balloons()
               st.write("thank's for keeping Twitter a Healthy Space!",colorsys="red" )
               st.success("There is Nothing Wrong in this tweet its Positive")

               return
           elif ans==1:

               st.error("There is Something Wrong in this tweet its Negative")
               return
    stop()