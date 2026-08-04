from flask import Flask, render_template, request
from src.model_loader import load_models
app = Flask(__name__)
movies, similarity = load_models()

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(movie):
    movie = movie.lower()

    # Find matching movie
    matches = movies[movies['title'].str.lower() == movie]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations

# -----------------------------
# Home Page
# -----------------------------
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

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)