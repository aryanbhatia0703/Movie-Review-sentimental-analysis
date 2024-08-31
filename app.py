import pandas as pd 
import re
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model = pk.load(open('model.pkl','rb'))
scaler = pk.load(open('scaler.pkl','rb'))
st.set_page_config(
    page_title="Movie review sentiment app",
    page_icon="🧊",
    layout='centered',
    initial_sidebar_state="auto"
)
st.title("Movie Review Sentiment predictor")
st.image("movie.png",width=700)
review = st.text_input('Enter Movie Review')

if st.button('Predict'):
    if re.match('^[a-zA-Z0-9@#\s]+$', review):
        review_scale = scaler.transform([review]).toarray()
        result = model.predict(review_scale)
        if result[0] == 0:
            st.write('Negative Review')
        else:
            st.write('Positive Review')
    else:
        st.write(" Input Not Valid")