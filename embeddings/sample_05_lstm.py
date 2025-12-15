from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import tensorflow as tf

# Sample data
text_data = ["my name is Ashwini", "my husband name is Pramod", "my surname name is waghmare", "i live in pune",
             "my native place is the latur", "i like mumbai"]

# 1. Data Preparation
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print(f"vocab_size: {vocab_size}")
input_sequences = []
for line in text_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    print(f"line: {line} \n token list {token_list}")
    for i in range(len(token_list), 1,-1):
        n_gram_sequence = token_list[:i + 1]
        reversed_sequence = n_gram_sequence[::-1]
        input_sequences.append(reversed_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
print(f"input_sequences: {input_sequences}")
padded_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post'))
print(f"padded_sequences:\n {padded_sequences}")

X, y = padded_sequences[:, 1:], padded_sequences[:, 0]
y = np.array(tf.keras.utils.to_categorical(y, num_classes=vocab_size))
print(f"X\n{X} \n y:\n{y}")

# 2. Model Architecture
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_sequence_len - 1))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

# 3. Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)  # Set verbose to 1 to see training progress


# 4. Inference (Auto-suggestion)
def generate_suggestions(seed_text, n_words=1):
    for _ in range(n_words):
        tokn_list = tokenizer.texts_to_sequences([seed_text])[0] #User query to embedding
        tokn_list = pad_sequences([tokn_list], maxlen=max_sequence_len - 1, padding='pre')
        print(model.predict(tokn_list))
        predicted = np.argmax(model.predict(tokn_list), axis=-1)
        print(predicted)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


print(generate_suggestions("is name my ", n_words=1))
# print(generate_suggestions("Ashwini", n_words=1))
