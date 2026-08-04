import os
import pickle
import pandas as pd
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i["name"])
    return L


def convert_cast(text):
    L = []
    count = 0
    for i in ast.literal_eval(text):
        if count < 3:
            L.append(i["name"])
            count += 1
    return L


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L


def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])


def generate_similarity():

    print("Generating similarity matrix...")

    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    credits = pd.read_csv(os.path.join(DATA_DIR, "credits.csv"))

    movies = movies.merge(credits, on="title")

    movies = movies[
        ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    ]

    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    movies["genres"] = movies["genres"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

    movies["keywords"] = movies["keywords"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

    movies["cast"] = movies["cast"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

    movies["crew"] = movies["crew"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[["movie_id", "title", "tags"]]

    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())
    new_df["tags"] = new_df["tags"].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words="english")

    vectors = cv.fit_transform(new_df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    os.makedirs(MODEL_DIR, exist_ok=True)

    pickle.dump(new_df, open(os.path.join(MODEL_DIR, "movie_list.pkl"), "wb"))
    pickle.dump(similarity, open(os.path.join(MODEL_DIR, "similarity.pkl"), "wb"))

    return new_df, similarity


def load_models():

    movie_file = os.path.join(MODEL_DIR, "movie_list.pkl")
    similarity_file = os.path.join(MODEL_DIR, "similarity.pkl")

    if os.path.exists(movie_file) and os.path.exists(similarity_file):

        print("Loading saved models...")

        movies = pickle.load(open(movie_file, "rb"))
        similarity = pickle.load(open(similarity_file, "rb"))

        return movies, similarity

    return generate_similarity()