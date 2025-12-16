# Skip-Gram predicts context words from a target word

# Skip-Gram training data

sentence = "machine learning is fun".split()

window_size = 1

pairs = []
for i, target in enumerate(sentence):
    for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
        if i != j:
            pairs.append((target, sentence[j]))

# print(pairs)

# Skip-Gram implementation (NumPy)
import numpy as np

# Vocabulary
vocab = list(set(sentence))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
V = len(vocab)

# Hyperparameters
embedding_dim = 5
learning_rate = 0.1
epochs = 500

# Initialize weights
W1 = np.random.randn(V, embedding_dim)   # input → embedding
W2 = np.random.randn(embedding_dim, V)   # embedding → output

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Training
for epoch in range(epochs):
    total_loss = 0

    for target_word, context_word in pairs:
        target_idx = word_to_idx[target_word]
        context_idx = word_to_idx[context_word]

        # Forward pass
        h = W1[target_idx]              # embedding of target
        u = np.dot(h, W2)
        y = softmax(u)

        total_loss -= np.log(y[context_idx])

        # Backprop
        e = y.copy()
        e[context_idx] -= 1

        W2 -= learning_rate * np.outer(h, e)
        W1[target_idx] -= learning_rate * np.dot(W2, e)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

