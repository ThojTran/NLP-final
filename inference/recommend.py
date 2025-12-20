"""Simple recommendation by topic-distribution similarity.

This module provides `recommend_articles`, a minimal helper that finds the
most similar documents to a query topic distribution using cosine similarity.
It returns a DataFrame with columns: ['title', 'category', 'similarity_score'].

The implementation is intentionally small and easy to explain for students.
"""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend_articles(
    query_topic_dist: Union[np.ndarray, list],
    corpus_topic_dists: Union[np.ndarray, list],
    metadata_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Recommend articles based on cosine similarity of topic distributions.

    Args:
        query_topic_dist: 1D array-like of topic probabilities for the query.
        corpus_topic_dists: 2D array-like with shape (n_docs, n_topics).
        metadata_df: DataFrame aligned with `corpus_topic_dists` and containing
            at least columns ['title', 'category'].
        top_k: number of recommendations to return.

    Returns:
        DataFrame with columns ['title', 'category', 'similarity_score'] sorted
        by decreasing similarity.

    Notes:
        - This is a simple academic example; no advanced filtering is applied.
    """
    # Convert inputs to numpy arrays
    q = np.asarray(query_topic_dist)
    docs = np.asarray(corpus_topic_dists)

    # Basic validation
    if q.ndim != 1:
        raise ValueError("query_topic_dist must be a 1D array-like")
    if docs.ndim != 2:
        raise ValueError("corpus_topic_dists must be a 2D array-like (n_docs, n_topics)")
    if docs.shape[1] != q.shape[0]:
        raise ValueError("Topic dimension mismatch between query and corpus")
    if len(metadata_df) != docs.shape[0]:
        raise ValueError("Length of metadata_df must match number of rows in corpus_topic_dists")

    # Compute cosine similarities
    sims = cosine_similarity(q.reshape(1, -1), docs).ravel()

    # Handle case where top_k > number of documents
    n_docs = sims.size
    k = min(top_k, n_docs)

    # Get top indices sorted by decreasing similarity
    top_indices = np.argsort(-sims)[:k]

    # Prepare result DataFrame
    results = metadata_df.iloc[top_indices].copy().reset_index(drop=True)
    results["similarity_score"] = sims[top_indices].astype(float)

    # Keep only requested columns in a readable order
    cols = [c for c in ["title", "category", "similarity_score"] if c in results.columns]
    return results[cols]
