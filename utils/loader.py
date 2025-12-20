"""Simple loader utilities for an LDA topic modeling project.

This module keeps things intentionally small and easy to explain for a
university assignment.

Functions:
- load_pickle(path): load and return an object from a pickle file
- load_models(model_dir): load 'lda_model.pkl' and 'count_vectorizer.pkl'
"""

import os
import pickle
from typing import Tuple


def load_pickle(path: str):
    """Load and return a Python object from a pickle file.

    Raises FileNotFoundError when the file does not exist. Other errors
    (e.g., unpickling errors) are propagated so students can see them.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def load_models(model_dir: str = "models") -> Tuple[object, object]:
    """Load LDA model and CountVectorizer from `model_dir`.

    Expects files named:
      - lda_model.pkl
      - count_vectorizer.pkl

    Returns a tuple: (lda_model, count_vectorizer)

    This function is intentionally minimal and will raise exceptions if
    files are missing or cannot be unpickled.
    """
    lda_path = os.path.join(model_dir, "lda_model.pkl")
    vec_path = os.path.join(model_dir, "count_vectorizer.pkl")

    lda_model = load_pickle(lda_path)
    count_vectorizer = load_pickle(vec_path)

    return lda_model, count_vectorizer
