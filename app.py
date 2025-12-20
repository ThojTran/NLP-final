import os
import streamlit as st
from typing import List

from topic_modeling_app.utils.loader import load_models
from topic_modeling_app.preprocessing.preprocessing import preprocess_single_text


def get_top_keywords(lda_model, vectorizer, topic_id: int, top_n: int = 10) -> List[str]:
    """Return top keywords (strings) for a given topic id."""
    feature_names = vectorizer.get_feature_names_out()
    topic_weights = lda_model.components_[topic_id]
    top_indices = topic_weights.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]


def main():
    st.title("Simple LDA Topic Demo — Vietnamese")
    st.write("Enter a short Vietnamese text and press **Predict Topic**.")

    # Load models (resolve models dir relative to package)
    pkg_dir = os.path.dirname(__file__)
    models_dir = os.path.join(pkg_dir, "models")

    try:
        lda_model, count_vectorizer = load_models(models_dir)
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return

    text = st.text_area("Input text", height=200)

    if st.button("Predict Topic"):
        if not text or not text.strip():
            st.warning("Please enter some text to predict.")
            return

        # Preprocess
        processed = preprocess_single_text(text)

        # Vectorize and predict
        doc_vec = count_vectorizer.transform([processed])
        topic_dist = lda_model.transform(doc_vec)[0]
        predicted = int(topic_dist.argmax())

        st.subheader("Prediction")
        st.write(f"**Predicted topic ID:** {predicted}")

        st.subheader("Top keywords for predicted topic")
        keywords = get_top_keywords(lda_model, count_vectorizer, predicted, top_n=10)
        st.write(", ".join(keywords))

        st.subheader("Topic distribution (top 5)")
        topk = sorted(enumerate(topic_dist), key=lambda x: x[1], reverse=True)[:5]
        for tid, prob in topk:
            # Generate a short title for the topic using top keywords (first 3)
            title_keywords = get_top_keywords(lda_model, count_vectorizer, tid, top_n=3)
            title = ", ".join(title_keywords)
            st.write(f"Topic {tid}: {prob:.4f} (title: {title})")


if __name__ == "__main__":
    main()