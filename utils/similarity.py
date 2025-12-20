"""Simple similarity helpers for topic modeling.

Functions are intentionally minimal and easy to explain for students.
"""

from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


def cosine_similarity_score(vec1, vec2) -> float:
    """Return cosine similarity between two vectors.

    Works with dense numpy arrays and scipy sparse vectors/matrices.
    Returns a float in [-1.0, 1.0].
    """
    # Prepare inputs as 2D (1, n_features) when needed
    def _as_2d(v):
        if sparse.issparse(v):
            return v
        arr = np.asarray(v)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    a = _as_2d(vec1)
    b = _as_2d(vec2)

    score = cosine_similarity(a, b)
    # cosine_similarity returns a matrix; we want the single value
    return float(score.ravel()[0])


def find_similar_documents(query_vector, document_vectors, top_k: int = 5) -> List[int]:
    """Return indices of the most similar documents to `query_vector`.

    Args:
        query_vector: single vector (1D array or sparse row)
        document_vectors: 2D array-like (n_docs, n_features) or sparse matrix
        top_k: number of top matches to return

    Returns:
        List of indices (ints) sorted by decreasing similarity.
    """
    # Ensure query is 2D for cosine computation
    if sparse.issparse(document_vectors):
        q = query_vector if sparse.issparse(query_vector) else np.asarray(query_vector).reshape(1, -1)
        sims = cosine_similarity(q, document_vectors).ravel()
    else:
        q = np.asarray(query_vector)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        docs = np.asarray(document_vectors)
        # If docs is 1D, make it 2D (single document)
        if docs.ndim == 1:
            docs = docs.reshape(1, -1)
        sims = cosine_similarity(q, docs).ravel()

    if sims.size == 0:
        return []

    k = min(top_k, sims.size)
    # argsort in descending order
    top_indices = np.argsort(-sims)[:k]
    return [int(i) for i in top_indices]
