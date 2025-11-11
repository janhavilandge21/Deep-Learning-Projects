import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

st.title("Movie Title Next-Word Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:/Users/JANHAVI/Downloads/tmdb_5000_movies.csv")
    titles = df['original_title'].tolist()
    return titles

movie_names = load_data()

# Tokenize and prepare sequences
@st.cache_resource
def prepare_data(movie_names):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(movie_names)
    seq = tokenizer.texts_to_sequences(movie_names)

    X = []
    Y = []

    for s in seq:
        if len(s) > 1:
            for i in range(1, len(s)):
                X.append(s[:i])
                Y.append(s[i])

    X = pad_sequences(X)
    Y = to_categorical(Y, num_classes=len(tokenizer.word_index)+1)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, X, Y, vocab_size

with st.spinner("Preparing data..."):
    tokenizer, X, Y, vocab_size = prepare_data(movie_names)

# Build improved model
@st.cache_resource
def build_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 32),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(vocab_size, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model(vocab_size)

if st.button("Train Model (Warning: Takes Time)"):
    with st.spinner("Training model..."):
        model.fit(X, Y, epochs=30, batch_size=64)
        model.save("movie_next_word_model.h5")
    st.success("Model Training Complete and Saved!")

# Prediction function
def make_prediction(text, n_words):
    vocab_array = np.array(list(tokenizer.word_index.keys()))
    for _ in range(n_words):
        token_seq = tokenizer.texts_to_sequences([text])
        token_padded = pad_sequences(token_seq, maxlen=X.shape[1])
        pred = np.argmax(model.predict(token_padded), axis=-1)[0]
        predicted_word = vocab_array[pred-1]
        text += " " + predicted_word
    return text

# UI
st.subheader("Generate Movie Title Continuation")
user_input = st.text_input("Enter starting word:")
num_words = st.slider("Number of words to generate:", 1, 10, 5)

if st.button("Generate"):
    if user_input.strip() != "":
        result = make_prediction(user_input, num_words)
        st.write("### Generated Title: ")
        st.success(result)
    else:
        st.warning("Please enter a valid word.")
