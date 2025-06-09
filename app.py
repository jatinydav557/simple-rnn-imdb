# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Streamlit App
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="wide")
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.markdown("Type a movie review below and click **Classify** to predict whether it's positive or negative.")

user_input = st.text_area('✍️ Your Movie Review Here:')

if st.button('🔍 Classify'):

    if user_input.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = '😊 Positive' if prediction[0][0] > 0.5 else '😞 Negative'
        confidence = float(prediction[0][0])

        # Sentiment Display
        st.subheader('🔎 Sentiment Result')
        st.markdown(f"<h2 style='color: {'green' if confidence > 0.5 else 'red'}'>{sentiment}</h2>", unsafe_allow_html=True)
        st.write(f"**Prediction Confidence:** {confidence:.2%}")

        # Confidence Bar
        st.progress(confidence if confidence > 0.5 else 1 - confidence)

        # Word Cloud
        st.subheader("☁️ Word Cloud of Your Review")
        generate_wordcloud(user_input)

else:
    st.info("Awaiting input... Enter a review above and click **Classify**.")

st.markdown("---")
st.caption("Built using TensorFlow and Streamlit")
