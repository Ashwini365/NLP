from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the documents and transform them into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a DataFrame for better readability
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print("TF-IDF Matrix:\n")
print(tfidf_df)
print(vectorizer.idf_)

# You can also get the TF-IDF vector for a specific document
# For example, for the first document:
print("\nTF-IDF vector for the first document:")
print(tfidf_df.iloc[0])
# tfidf_df.to_csv("tfidf_matrix.csv")