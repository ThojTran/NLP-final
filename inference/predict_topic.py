"""Simple prediction helper for LDA topic modeling.

This module provides a single function `predict_topic` that:
1. Preprocesses the input text using a provided preprocessor
2. Vectorizes it with a fitted CountVectorizer
3. Uses a fitted scikit-learn LDA model to get topic distribution

The code is intentionally minimal and easy to explain for a university
assignment.
"""

import os
from typing import Tuple, List
from topic_modeling_app.utils.loader import load_models


def predict_topic(text: str, preprocessor, lda_model, vectorizer) -> Tuple[int, List[float]]:
    """Predict the most likely topic for a single text.

    Args:
        text: raw input text (string)
        preprocessor: either a callable that accepts text and returns a
            preprocessed string, or an object with a `preprocess_text` method
        lda_model: fitted sklearn.decomposition.LatentDirichletAllocation
        vectorizer: fitted sklearn.feature_extraction.text.CountVectorizer

    Returns:
        (predicted_topic_id, topic_distribution)
        - predicted_topic_id: int (index of topic with highest probability)
        - topic_distribution: list of floats (length = n_topics)
    """
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # Apply preprocessor
    if callable(preprocessor):
        processed = preprocessor(text)
    elif hasattr(preprocessor, "preprocess_text"):
        processed = preprocessor.preprocess_text(text)
    else:
        raise ValueError("preprocessor must be callable or have preprocess_text method")

    # Vectorize (expects a list-like of documents)
    doc_vector = vectorizer.transform([processed])

    # Get topic distribution from LDA
    topic_dist = lda_model.transform(doc_vector)[0]

    # Convert to Python list of floats and get argmax
    topic_distribution = [float(x) for x in topic_dist]
    predicted_topic_id = int(topic_distribution.index(max(topic_distribution)))

    return predicted_topic_id, topic_distribution


if __name__ == "__main__":
    # Simple example demonstrating how to load models and call predict_topic
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(pkg_dir, "models")
    lda_model, count_vectorizer = load_models(models_dir)

    try:
        # Import a simple preprocessing helper from the project (used in training)
        from topic_modeling_app.preprocessing.preprocessing import preprocess_single_text
        preprocessor_fn = preprocess_single_text
    except Exception:
        # Fallback: identity function (not recommended for real use)
        preprocessor_fn = lambda x: x

    sample_text = """Dự báo thời tiết hôm nay cho biết khu vực miền Bắc sẽ có mưa rào và dông 
            vào chiều tối. Nhiệt độ dao động từ 22-28 độ C. Khu vực Đông Bắc có gió 
            mạnh cấp 3-4. Thời tiết miền Nam nắng nóng với nhiệt độ cao nhất 35 độ. 
            Các tỉnh miền Trung trời rét về đêm và sáng sớm. Cơ quan khí tượng khuyến 
            cáo người dân đề phòng mưa dông và gió giật mạnh. Biển Đông có gió Đông Bắc 
            cấp 5-6."""
    topic_id, distribution = predict_topic(sample_text, preprocessor_fn, lda_model, count_vectorizer)
    print(f"Predicted topic: {topic_id}")
    print(f"Top 5 probs: {distribution[:5]}")
