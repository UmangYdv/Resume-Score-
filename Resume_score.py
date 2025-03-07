from pdfminer.high_level import extract_text
import os
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

# Extract text from PDF
def extract_resume_text(pdf_path):
    text = extract_text(pdf_path)
    return text

# Clean resume text
def clean_resume(text):
    text = re.sub(r'\n+', ' ', text)  # Remove newlines
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    return ' '.join(words)

# Example usage
resume_text = extract_resume_text("path")
cleaned_resume = clean_resume(resume_text)

# Load sample dataset
df = pd.read_csv("resume_dataset_path")  # Contains columns: "Resume", "Category"

# Convert text data into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Resume"])

# Encode job categories
encoder = LabelEncoder()
y = encoder.fit_transform(df["Category"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Rank resumes
def rank_resumes(job_description, resumes):
    job_vector = vectorizer.transform([job_description])
    resume_vectors = vectorizer.transform(resumes)
    scores = cosine_similarity(job_vector, resume_vectors).flatten()
    return scores

# Example job description
job_desc = "Looking for a Data Scientist with experience in Python, Machine Learning, and Deep Learning."

# Rank all resumes in the DataFrame
resume_texts = df["Resume"]  # Use all resumes, not just the first 10
scores = rank_resumes(job_desc, resume_texts)

# Assign scores to the DataFrame
df["match_score"] = scores

# Sort the DataFrame by match score
df_sorted = df.sort_values(by="match_score", ascending=False)
print(df_sorted[["Category", "match_score"]])

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    job_desc = request.form["job_description"]
    resume_text = request.form["resume_text"]
    score = rank_resumes(job_desc, [resume_text])[0]
    return f"Resume Match Score: {score:.2f}"

if __name__ == '__main__':
    app.run(debug=True)