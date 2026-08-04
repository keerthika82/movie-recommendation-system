import pandas as pd
import ast
import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

movies_path = os.path.join(BASE_DIR, "data", "movies.csv")
credits_path = os.path.join(BASE_DIR, "data", "credits.csv")
model_path = os.path.join(BASE_DIR, "model")

# -----------------------------
# Load datasets
# -----------------------------
movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

movies = movies.merge(credits, on="title")

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.dropna(inplace=True)

# -----------------------------
# Helper Functions
# -----------------------------
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def stem(text):
    y = []
    for word in text.split():
        y.append(ps.stem(word))
    return " ".join(y)

# -----------------------------
# Data Processing
# -----------------------------
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
new_df['tags'] = new_df['tags'].apply(stem)

# -----------------------------
# Vectorization
# -----------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

# -----------------------------
# Similarity
# -----------------------------
similarity = cosine_similarity(vectors)

# -----------------------------
# Save Models
# -----------------------------
os.makedirs(model_path, exist_ok=True)

pickle.dump(new_df, open(os.path.join(model_path, "movie_list.pkl"), "wb"))
pickle.dump(similarity, open(os.path.join(model_path, "similarity.pkl"), "wb"))

print("===================================")
print(" Recommendation Model Created!")
print("===================================")

print("\nMovies:", len(new_df))

print("Similarity Matrix Shape:", similarity.shape)

print("\nFiles Saved Successfully!")

print("movie_list.pkl")
print("similarity.pkl")