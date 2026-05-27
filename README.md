# 🔭 Redshift Predictor

A machine learning web application that predicts galaxy redshift values from spectral data using two models — Random Forest and a Neural Network (MLP). Built as a final-year astronomy project.

---

## Overview

This app takes `.npy` spectral and metadata files as input, applies PCA for dimensionality reduction, trains two regression models, and displays prediction plots alongside key metrics — all through a browser-based interface.

It also includes a simple AI chatbot backed by a local knowledge base (`astro.txt`) to answer astronomy and ML-related questions.

---

## Features

- Upload `.npy` spectral and metadata files via the web UI
- Dimensionality reduction with PCA (2 components)
- Two trained models:
  - **Random Forest Regressor** (100 estimators)
  - **Neural Network / MLP Regressor** (1 hidden layer, 50 units)
- Prediction vs. actual scatter plots (base64-encoded, rendered in browser)
- Metrics: R² score, MSE, sample counts, redshift statistics
- Built-in astronomy chatbot powered by `astro.txt` knowledge base

---

## Project Structure

```
redshift-predictor/
├── app.py            # Flask backend — ML pipeline, API routes, chatbot logic
├── index.html        # Frontend UI
├── astro.txt         # Knowledge base for the chatbot
├── requirements.txt  # Python dependencies
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Peace5000/redshift-predictor.git
cd redshift-predictor
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

Then open your browser and go to `http://localhost:5000`.

---

## Usage

1. Upload your spectral data file (`.npy`) and metadata file (`.npy`) using the web interface.
2. Click **Predict** to run the ML pipeline.
3. View results: model accuracy, R², MSE, and prediction plots.
4. Use the **Chat** feature to ask questions about redshift or the models.

### Expected Data Format

| File | Shape | Description |
|------|-------|-------------|
| `spectra.npy` | `(n_samples, n_wavelengths)` | Spectral flux values per galaxy |
| `metadata.npy` | `(n_samples, ...)` | First column must be the redshift value (`z`) |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask, Flask-CORS |
| ML | scikit-learn (PCA, RandomForest, MLP, StandardScaler) |
| Visualisation | Matplotlib |
| Frontend | HTML, CSS, JavaScript |
| Server | Gunicorn (for production) |

---

## Dependencies

```
Flask==3.0.0
flask-cors==4.0.0
numpy==1.24.3
matplotlib==3.7.1
scikit-learn==1.3.0
gunicorn==21.2.0
```

---

## Chatbot

The built-in chatbot reads from `astro.txt` at startup and answers questions using keyword-based search. To customise it, edit `astro.txt` with any astronomy or ML content you'd like it to reference.

---

## Author

**Peace Khutso** — Final Year Machine Learning Astronomy Project
