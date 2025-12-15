import re
from gensim.models import FastText

# ---------------------------
# 1. Your own documents
# ---------------------------
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# ---------------------------
# 2. Preprocessing
# ---------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()

sentences = [preprocess(doc) for doc in documents]

print("Tokenized sentences:")
print(sentences)

# ---------------------------
# 3. Train FastText
# ---------------------------
model = FastText(
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1  # 1 = skip-gram, 0 = CBOW
)

model.build_vocab(sentences)
model.train(sentences, total_examples=len(sentences), epochs=10)

# ---------------------------
# 4. Use the model
# ---------------------------
print("\nVector for 'document':")
print(model.wv['document'])

print("\nMost similar to 'document':")
print(model.wv.most_similar('document'))

print("\nSimilarity between 'first' and 'second':")
print(model.wv.similarity('first', 'second'))

print("\nVector for unseen word 'documentsssss':")
print(model.wv['documentsssss'])

# ---------------------------
# 5. Save + Load
# ---------------------------
model.save("my_fasttext.model")
print("\nModel saved!")

loaded_model = FastText.load("my_fasttext.model")
print("Model loaded!")

