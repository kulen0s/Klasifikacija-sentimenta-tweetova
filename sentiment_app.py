import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 🔹 Učitaj spremljeni model
model = tf.keras.models.load_model('sentiment_lstm_glove_model.h5')

# 🔹 Učitaj spremljeni tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 🔹 Definiraj iste parametre kao kod treniranja
max_len = 60  # mora biti isto kao u treniranju!

# 🔹 Streamlit sučelje
st.title("Sentiment Analysis App")
st.write("Enter a sentence to determine its sentiment (positive or negative).")

# 🔹 Unos korisnika
user_input = st.text_input("Enter your text:")

if user_input:
    # Tokenizacija i padding
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_pad = pad_sequences(input_seq, maxlen=max_len, padding='post')

    # Predikcija
    prediction = model.predict(input_pad)
    sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞"

    # Prikaz rezultata
    st.write(f"**Predicted Sentiment:** {sentiment}")
    st.write(f"**Prediction Score:** {prediction[0][0]:.4f}")