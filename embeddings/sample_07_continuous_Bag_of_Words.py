import numpy as np

# Sentence
sentence = "machine learning is fun".split()

# Vocabulary
vocab = list(set(sentence))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
V = len(vocab)

# Hyperparameters
embedding_dim = 5
learning_rate = 0.1
epochs = 500
window = 1

# Generate CBOW data
def cbow_data(words, window):
    pairs = []
    for i in range(window, len(words) - window):
        context = [words[i - 1], words[i + 1]]
        target = words[i]
        pairs.append((context, target))
    return pairs

data = cbow_data(sentence, window)

# Initialize weights
W1 = np.random.randn(V, embedding_dim)
W2 = np.random.randn(embedding_dim, V)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Training
for epoch in range(epochs):
    total_loss = 0
    for context, target in data:
        context_idx = [word_to_idx[w] for w in context]
        target_idx = word_to_idx[target]

        # Forward
        h = np.mean(W1[context_idx], axis=0)
        u = np.dot(h, W2)
        y = softmax(u)

        total_loss -= np.log(y[target_idx])

        # Backprop
        e = y
        e[target_idx] -= 1

        W2 -= learning_rate * np.outer(h, e)
        for idx in context_idx:
            W1[idx] -= learning_rate * np.dot(W2, e) / len(context_idx)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

for word in vocab:
    print(word, W1[word_to_idx[word]])

