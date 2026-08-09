from flask import Flask, render_template, request, jsonify

from src.model_loader import load_models

from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# Load movie data and vectors
movies, vectors = load_models()


# --------------------------------
# Recommendation Function
# --------------------------------

def recommend(movie):

    movie = movie.lower().strip()

    matches = movies[
        movies["title"].str.lower() == movie
    ]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    similarity_scores = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

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
# API for GitHub Pages
# --------------------------------

@app.route("/api/recommend", methods=["POST"])
def api_recommend():

    data = request.get_json()

    if not data or "movie" not in data:

        return jsonify({
            "recommendations": [],
            "error": "Movie name is required"
        }), 400

    movie = data["movie"]

    recommendations = recommend(movie)

    response = jsonify({
        "recommendations": recommendations
    })

    return response


# --------------------------------
# Allow GitHub Pages to access API
# --------------------------------

@app.after_request
def add_cors_headers(response):

    response.headers["Access-Control-Allow-Origin"] = "*"

    response.headers["Access-Control-Allow-Headers"] = "Content-Type"

    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"

    return response


# --------------------------------
# Run App
# --------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
