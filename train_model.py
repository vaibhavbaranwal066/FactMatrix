#!/usr/bin/env python3
"""
Train an enhanced fake-news detector using TF-IDF + Linear SVM (SGDClassifier).
This version uses the ISOT benchmark dataset, incorporates advanced text
preprocessing, and uses GridSearchCV for hyperparameter tuning to maximize performance.

Saves the best-performing pipeline with joblib.

Usage:
    python train_model.py --data_dir data/ --out models/factmatrix_v2_clf.joblib
"""
import os
import re
import joblib
import argparse
import warnings
import string

import pandas as pd
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")


def load_isot_dataset(data_dir: str) -> pd.DataFrame:
    """Loads and combines the ISOT Fake News Dataset (True.csv & Fake.csv)."""
    true_path = os.path.join(data_dir, 'True.csv')
    fake_path = os.path.join(data_dir, 'Fake.csv')

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(
            f"ISOT dataset files ('True.csv', 'Fake.csv') not found in: {data_dir}"
        )

    # Some CSVs might contain odd encodings; latin-1 is safer for unknowns
    df_true = pd.read_csv(true_path, encoding='latin-1', engine='python')
    df_fake = pd.read_csv(fake_path, encoding='latin-1', engine='python')

    # Assign labels before combining
    df_true['label'] = 'REAL'
    df_fake['label'] = 'FAKE'

    # Combine into a single dataframe
    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Ensure the expected columns exist; if not, try alternative names
    if 'title' not in df.columns or 'text' not in df.columns:
        raise ValueError("Expected 'title' and 'text' columns in ISOT CSVs.")

    # Combine title and text for a richer feature set
    df['text'] = df['title'].astype(str) + " " + df['text'].astype(str)
    df = df[['text', 'label']]  # Keep only necessary columns

    print("ISOT Dataset loaded and combined successfully.")
    print(f"Total samples: {len(df)}")
    print(df.label.value_counts())

    return df


def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """Advanced text cleaning and normalization."""
    # 1. Convert to lowercase
    text = str(text).lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Remove numbers
    text = re.sub(r'\d+', '', text)
    # 4. Remove extra whitespace
    text = " ".join(text.split())
    # 5. Tokenize, lemmatize, and remove stopwords (keep words length > 2)
    words = [
        lemmatizer.lemmatize(word) for word in text.split()
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)


def build_pipeline() -> Pipeline:
    """Builds a TF-IDF + linear SVM (SGD hinge) pipeline for GridSearchCV."""
    # We already lowercase in preprocess_text, so set lowercase=False here.
    # If you change preprocess_text to not lowercase, set lowercase=True.
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b',
            strip_accents='unicode',
            lowercase=False,  # we've already lowercased during preprocessing
            max_features=None,  # allow GridSearch to tune other params
        )),
        ('clf', SGDClassifier(
            loss='hinge',        # linear SVM
            penalty='l2',
            max_iter=2000,
            tol=1e-4,
            random_state=42,
            class_weight='balanced'  # helpful when dataset is imbalanced
        ))
    ])
    return pipeline


def main(data_dir: str, out_path: str):
    # Ensure NLTK resources are available. If not, download them.
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("Downloading NLTK omw-1.4...")
        nltk.download('omw-1.4', quiet=True)

    df = load_isot_dataset(data_dir)

    print("\nPreprocessing text data... (this may take a while)")
    # Prepare lemmatizer and stopwords once for speed
    lemmatizer = WordNetLemmatizer()
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        # If stopwords aren't found for some reason, fallback to a small set
        stop_words = {"the", "and", "is", "in", "it", "of", "to", "a"}
        print("Warning: using fallback minimal stopword set.")

    # Apply preprocessing
    df['text'] = df['text'].astype(str).fillna("").apply(
        lambda t: preprocess_text(t, lemmatizer, stop_words)
    )

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()

    # Parameter grid for GridSearchCV
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_df': [0.8, 0.9],
        'tfidf__min_df': [1, 2],
        'clf__alpha': [1e-5, 1e-4, 1e-3],
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=2)

    print("\nStarting hyperparameter tuning with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters set found:")
    print(grid_search.best_params_)

    best_pipe = grid_search.best_estimator_

    # Evaluation
    y_pred = best_pipe.predict(X_test)
    print("\n=== Classification Report on Test Set ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Save the best model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(best_pipe, out_path)
    print(f"\n✅ Best model saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train enhanced fake news detector (TF-IDF + Linear SVM with GridSearchCV)"
    )
    parser.add_argument("--data_dir", default="data/", help="Path to directory containing ISOT dataset (True.csv and Fake.csv)")
    parser.add_argument("--out", default="models/factmatrix_v2_clf.joblib", help="Output joblib model path")
    args = parser.parse_args()
    main(args.data_dir, args.out)