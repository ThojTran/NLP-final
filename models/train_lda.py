"""Train a simple LDA model and save the model + CountVectorizer.

This script is intentionally minimal and easy to follow for a university
assignment. It reads `data/processed_data.csv` and uses the
`combined_text` column for training.
"""

import os
import pickle
import numpy as np

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def main():
    # Paths (keep simple and relative to this file)
    script_dir = os.path.dirname(__file__)
    # Use the preprocessed data produced by data/run_preprocessing.py
    data_path = os.path.abspath(os.path.join(script_dir, "..", "data", "processed_data.csv"))
    models_dir = script_dir  # save into the models/ directory

    # Basic checks
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Load processed data
    df = pd.read_csv(data_path, encoding="utf-8")
    if "combined_text" not in df.columns:
        raise ValueError("Expected column 'combined_text' in processed_data.csv")

    texts = df["combined_text"].astype(str).tolist()
    print(f"Loaded {len(texts)} documents for training")

    # Vectorizer hyperparameters (simple and explainable)
    min_df = 5                # ignore tokens that appear in fewer than 5 documents
    max_df = 0.9              # ignore tokens that appear in more than 90% of docs
    max_features = 10000      # keep top 10k tokens by frequency
    ngram_range = (1, 2)      # consider unigrams and bigrams

    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    print(f"Created document-term matrix with shape {X.shape}")

    # Train LDA with slightly more iterations and batch learning
    n_topics = 10  # fixed and simple to explain; you can try other values manually
    lda_max_iter = 30
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=lda_max_iter, learning_method='batch', random_state=42)
    print(f"Training LDA with {n_topics} topics (max_iter={lda_max_iter}, batch)...")
    lda.fit(X)
    print("LDA training completed")

    # Inspect topic sharpness (average peak topic probability)
    dists = lda.transform(X)
    avg_peak = float(dists.max(axis=1).mean())
    print(f"Average peak topic probability (higher is sharper): {avg_peak:.4f}")

    # Print top keywords per topic (helpful for manual inspection)
    feature_names = vectorizer.get_feature_names_out()
    top_n = 10
    print("Top keywords per topic:")
    for t in range(n_topics):
        top_idx = lda.components_[t].argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_idx]
        print(f"Topic {t}: {', '.join(top_words)}")

    # Save models
    lda_path = os.path.join(models_dir, "lda_model.pkl")
    vec_path = os.path.join(models_dir, "count_vectorizer.pkl")

    with open(lda_path, "wb") as f:
        pickle.dump(lda, f)

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved LDA model to: {lda_path}")
    print(f"Saved CountVectorizer to: {vec_path}")


if __name__ == "__main__":
    main()