# 🎬 Movie Recommendation System

A professional Movie Recommendation System built using **Python**, **Flask**, and **Machine Learning (Content-Based Filtering)**.

This application recommends movies based on their similarity using genres, keywords, cast, director, and movie overview.

---

## 📌 Features

- 🎥 Movie Recommendation
- 🔍 Select a movie from the dropdown
- 🤖 Content-Based Filtering
- 📊 Cosine Similarity Algorithm
- 🌐 Flask Web Application
- 🎨 Responsive User Interface
- ⚡ Fast Recommendations

---

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML5
- CSS3
- JavaScript
- Git
- GitHub

---

## 📂 Project Structure

```text
movie-recommendation-system/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
├── model/
├── src/
├── static/
└── templates/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
```

Move into the project:

```bash
cd movie-recommendation-system
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 🧠 Machine Learning Workflow

1. Load Movie Dataset
2. Merge Movie and Credits Dataset
3. Data Cleaning
4. Feature Engineering
5. Count Vectorization
6. Cosine Similarity
7. Recommendation Generation

---

## 📸 Screenshots

### Home Page

_Add a screenshot here_

### Recommendations

_Add a screenshot here_

---

## 🚀 Future Improvements

- Movie Posters
- Search Suggestions
- User Login
- Favorites
- Ratings
- Collaborative Filtering
- TMDB API Integration
- Deployment on Render

---

## 👨‍💻 Author

**Shahid**

GitHub:
https://github.com/YOUR_USERNAME

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.
## Generate the Recommendation Model

After cloning the repository, generate the recommendation model by running:

```bash
python src/preprocess.py
```

This creates the required files inside the `model/` folder.
