
import numpy as np
from collections import defaultdict

# Prepare data
corpus = [
    "i like deep learning",
    "i like nlp",
    "i enjoy learning"
]

tokens = [sentence.split() for sentence in corpus]
vocab = sorted(set(word for sent in tokens for word in sent))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
V = len(vocab)

window_size = 1

# Compute co-occurrence counts
X = defaultdict(float)

for sentence in tokens:
    for i, word in enumerate(sentence):
        wi = word_to_idx[word]
        for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
            if i != j:
                wj = word_to_idx[sentence[j]]
                X[(wi, wj)] += 1.0

# GloVe model implementation
embedding_dim = 5
learning_rate = 0.05
epochs = 50
x_max = 10
alpha = 0.75

W = np.random.randn(V, embedding_dim)
W_tilde = np.random.randn(V, embedding_dim)
b = np.zeros(V)
b_tilde = np.zeros(V)


def weighting(x):
    return (x / x_max) ** alpha if x < x_max else 1.0
for epoch in range(epochs):
    total_loss = 0

    for (i, j), x_ij in X.items():
        w_i = W[i]
        w_j = W_tilde[j]

        f_x = weighting(x_ij)
        diff = np.dot(w_i, w_j) + b[i] + b_tilde[j] - np.log(x_ij)

        loss = f_x * diff ** 2
        total_loss += loss

        # Gradients
        grad = 2 * f_x * diff

        W[i] -= learning_rate * grad * w_j
        W_tilde[j] -= learning_rate * grad * w_i
        b[i] -= learning_rate * grad
        b_tilde[j] -= learning_rate * grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        embeddings = W + W_tilde

    for word in vocab:
        print(word, embeddings[word_to_idx[word]])


