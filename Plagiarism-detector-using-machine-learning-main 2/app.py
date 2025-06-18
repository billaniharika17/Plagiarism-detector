from flask import Flask, render_template, request, jsonify
import pickle
import os
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load ML model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Load Hugging Face paraphrasing model (T5-based)
hf_model_name = "ramsrigouthamg/t5_paraphraser"
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)

# Get confidence level label
def get_confidence_level(percentage):
    if percentage >= 75:
        return "High"
    elif percentage >= 50:
        return "Moderate"
    elif percentage >= 25:
        return "Low"
    return "Very Low"

# Detect plagiarism and return percentage
def detect_with_percentage(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vectorized_text)[0]
        plagiarism_percentage = proba[1] * 100
    else:
        prediction = model.predict(vectorized_text)[0]
        plagiarism_percentage = 100.0 if prediction == 1 else 0.0
    is_plagiarized = plagiarism_percentage > 50
    return is_plagiarized, plagiarism_percentage

# Rewrite text using Hugging Face model
def rewrite_with_huggingface(text):
    try:
        input_text = f"paraphrase: {text} </s>"
        encoding = tokenizer.encode_plus(input_text, padding="max_length", return_tensors="pt", max_length=256, truncation=True)
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outputs = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            num_beams=5,
            early_stopping=True
        )

        rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rewritten_text
    except Exception as e:
        logging.error(f"Rewrite error: {str(e)}")
        return f"Error rewriting: {str(e)}"

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Detect plagiarism and optionally rewrite
@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    try:
        input_text = request.form['text']
        is_plagiarized, percentage = detect_with_percentage(input_text)
        rewritten_text = rewrite_with_huggingface(input_text) if is_plagiarized else None

        return render_template('index.html',
                               result=f"Plagiarism Detected: {percentage:.1f}%" if is_plagiarized else "No Plagiarism Detected",
                               percentage=percentage,
                               confidence_level=get_confidence_level(percentage),
                               rewritten_text=rewritten_text)
    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        return render_template('index.html', result="An unexpected error occurred.")

# API version of the plagiarism checker
@app.route('/api/detect', methods=['POST'])
def api_detect():
    data = request.json
    input_text = data.get('text', '')
    is_plagiarized, percentage = detect_with_percentage(input_text)

    response = {
        'is_plagiarized': is_plagiarized,
        'percentage': percentage,
        'confidence': get_confidence_level(percentage)
    }

    if is_plagiarized:
        response['rewritten_text'] = rewrite_with_huggingface(input_text)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
