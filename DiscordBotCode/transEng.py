from keras.models import load_model
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, Dropout, LSTM, TimeDistributed, Bidirectional
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import pickle


def text_to_sequences(tokenizer, text):
    return tokenizer.texts_to_sequences(text)
def sequence_padding(sequences, length=None):
    return pad_sequences(sequences, maxlen=length, padding='post')
max_length = 12


def translate_french_to_english(input_text):
    # Load the saved model and tokenizers
    model = load_model('translation_model4.h5')
    with open('eng_tokenizer4.pkl', 'rb') as handle:
        eng_tokenizer = pickle.load(handle)
    with open('fr_tokenizer4.pkl', 'rb') as handle:
        fr_tokenizer = pickle.load(handle)

    # Preprocess the input text
    sequences = text_to_sequences(fr_tokenizer, [input_text])
    padded_sequences = sequence_padding(sequences, max_length)

    # Make a prediction
    prediction = model.predict(padded_sequences)
    predicted_sequence = np.argmax(prediction, axis=-1)[0]

    # Reverse word indices for both languages
    reverse_fr_word_index = dict(map(reversed, fr_tokenizer.word_index.items()))
    reverse_eng_word_index = dict(map(reversed, eng_tokenizer.word_index.items()))

    # Convert the predicted sequence to text with fallback to French word
    translated_text = []
    for idx, word_idx in enumerate(padded_sequences[0]):
        if word_idx > 0:  # Ignore padding
            french_word = reverse_fr_word_index.get(word_idx, '')
            translated_word = reverse_eng_word_index.get(predicted_sequence[idx], french_word)
            translated_text.append(translated_word)

    return ' '.join(translated_text)


