import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'The cat sat on the mat',
    'The dog ran in the park'
]
# Initialize the vectorizer
vectorizer = CountVectorizer(lowercase=True)

# Fit and transform the corpus into a matrix
X = vectorizer.fit_transform(corpus)

# Convert the sparse matrix to a dense array for viewing
count_matrix = X.toarray()

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Display as a DataFrame for clarity
df = pd.DataFrame(data=count_matrix, columns=feature_names, index=['Doc 1', 'Doc 2'])
print(df)