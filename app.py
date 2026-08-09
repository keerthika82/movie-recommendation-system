from flask import Flask, render_template, request

from src.model_loader import load_models

from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# Load movie data and vectors
movies, vectors = load_models()


# --------------------------------
# Recommendation Function
# --------------------------------

def recommend(movie):

    movie = movie.lower()

    # Find matching movie
    matches = movies[
        movies["title"].str.lower() == movie
    ]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    # Calculate similarity only for the selected movie
    similarity_scores = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    # Get top 5 movies
    movie_indices = similarity_scores.argsort()[-6:][::-1]

    recommendations = []

    for index in movie_indices:

        if index != movie_index:
            recommendations.append(
                movies.iloc[index]["title"]
            )

        if len(recommendations) == 5:
            break

    return recommendations


# --------------------------------
# Home Page
# --------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    if request.method == "POST":

        movie = request.form["movie"]

        recommendations = recommend(movie)

    movie_names = movies["title"].values

    return render_template(
        "index.html",
        movie_names=movie_names,
        recommendations=recommendations
    )


# --------------------------------
# Run App
# --------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
