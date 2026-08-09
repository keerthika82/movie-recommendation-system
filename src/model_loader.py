import os
import pandas as pd
import ast

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def load_models():

    print("Loading movie data...")

    movies = pd.read_csv(
        os.path.join(DATA_DIR, "movies.csv")
    )

    credits = pd.read_csv(
        os.path.join(DATA_DIR, "credits.csv")
    )

    print("Merging movie data...")

    movies = movies.merge(
        credits,
        on="title"
    )

    movies = movies[
        [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew"
        ]
    ]

    movies.dropna(inplace=True)

    print("Processing movie information...")

    movies["genres"] = movies["genres"].apply(convert)

    movies["keywords"] = movies["keywords"].apply(convert)

    movies["cast"] = movies["cast"].apply(convert_cast)

    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].apply(
        lambda x: x.split()
    )

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

    movies = movies[
        ["movie_id", "title", "tags"]
    ].copy()

    movies["tags"] = movies["tags"].apply(
        lambda x: " ".join(x).lower()
    )

    movies["tags"] = movies["tags"].apply(stem)

    print("Creating movie vectors...")

    cv = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = cv.fit_transform(
        movies["tags"]
    )

    print("Movie vectors created successfully.")

    return movies, vectors
