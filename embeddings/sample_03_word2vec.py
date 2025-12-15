from gensim.models import Word2Vec
from gensim.test.utils import common_texts # A small sample corpus for demonstration

# 1. Prepare your corpus
# In a real-world scenario, 'sentences' would be a list of lists of words (tokenized sentences)
# For this example, we use common_texts from gensim, which is a list of tokenized sentences.
sentences = common_texts

# 2. Initialize and train the Word2Vec model
# vector_size: Dimension of the word embeddings
# window: Maximum distance between the current and predicted word within a sentence
# min_count: Ignores all words with total frequency lower than this
# workers: Number of worker threads to use for training
# sg: 0 for CBOW (Continuous Bag of Words), 1 for Skip-gram (default is CBOW)
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)

# 3. Access word vectors
# You can get the vector for a specific word
word_vector = model.wv['human']
print(f"Vector for 'human':\n{word_vector}\n")

# 4. Find most similar words
# Find words most similar to a given word based on their vector proximity
similar_words = model.wv.most_similar('human')
print(f"Words most similar to 'human':\n{similar_words}\n")

# 5. Calculate similarity between two words
similarity_score = model.wv.similarity('human', 'interface')
print(f"Similarity between 'human' and 'interface': {similarity_score}\n")

# 6. Save and load the model (optional)
# Saving the trained model
# model.save("word2vec_model.model")
# print("Model saved as 'word2vec_model.model'\n")

# Loading the saved model
# loaded_model = Word2Vec.load("word2vec_model.model")
# print("Model loaded successfully.\n")

# You can then use the loaded model for the same operations
# print(f"Vector for 'system' from loaded model:\n{loaded_model.wv['system']}\n")