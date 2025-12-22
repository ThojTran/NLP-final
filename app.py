import os
import streamlit as st
from typing import List

import pandas as pd

from topic_modeling_app.utils.loader import load_models
from topic_modeling_app.preprocessing.preprocessing import preprocess_single_text


def get_top_keywords(lda_model, vectorizer, topic_id: int, top_n: int = 10) -> List[str]:
    """Trả về danh sách từ khóa (chuỗi) cho một topic id."""
    feature_names = vectorizer.get_feature_names_out()
    topic_weights = lda_model.components_[topic_id]
    top_indices = topic_weights.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]


def _join_vietnamese(items: List[str]) -> str:
    """Join list into Vietnamese style: a, b và c"""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} và {items[1]}"
    return ", ".join(items[:-1]) + f" và {items[-1]}"


def generate_caption_from_input(keywords: List[str], processed_input: str, num_keywords: int = 3) -> str:
    """
    Tạo caption (Đề tài) dựa trên các từ khóa xuất hiện trong input đã tiền xử lý.
    Giữ nguyên hành vi trước đó.
    """
    if not keywords:
        return ""
    input_tokens = set(processed_input.split())
    matched = [k for k in keywords if k in input_tokens]
    if matched:
        top = matched[:num_keywords]
        readable = [t.replace("_", " ") for t in top]
        joined = _join_vietnamese(readable)
        return f"Đề tài: {joined}."
    else:
        top = keywords[:num_keywords]
        readable = [t.replace("_", " ") for t in top]
        joined = _join_vietnamese(readable)
        return f"Đề tài (gợi ý): {joined}."


def generate_topic_title(keywords: List[str], top_n: int = 8) -> str:
    """
    Sinh nhãn chủ đề ngắn (categorical label), ví dụ "Bóng đá Nữ", "Kinh tế".
    Giữ nguyên quy tắc đơn giản hiện có.
    """
    if not keywords:
        return "Không xác định"

    readable = [k.replace("_", " ").lower() for k in keywords[:top_n]]
    joined = " ".join(readable)
    tokens = set()
    for r in readable:
        tokens.update(r.split())

    sports = {"bóng", "bóng đá", "trận", "đội", "vđv", "thể thao", "trận đấu", "vòng"}
    politics = {"chính", "chính phủ", "đảng", "trung ương", "tỉnh", "huyện", "ủy", "quốc hội", "chính quyền"}
    economy = {"doanh", "doanh nghiệp", "giá", "usd", "đồng", "tỷ", "vốn", "kinh tế", "đầu tư", "thị trường"}
    weather = {"mưa", "nắng", "gió", "thời tiết", "nhiệt độ", "bão"}
    health = {"bệnh", "dịch", "tiêm", "sức khỏe", "covid", "vắc xin"}
    traffic = {"giao thông", "dự án", "cầu", "đường", "xe", "tàu", "ô tô"}
    crime = {"công an", "án", "tội", "bắt", "vụ", "khởi tố"}
    culture = {"văn hóa", "lễ", "lễ hội", "nghệ", "dịch vụ"}
    education = {"học", "trường", "sinh viên", "giảng viên", "giáo viên"}
    tech = {"công nghệ", "app", "web", "ai", "robot"}

    categories = [
        ("Bóng đá / Thể thao", sports),
        ("Chính trị", politics),
        ("Kinh tế", economy),
        ("Thời tiết", weather),
        ("Sức khỏe", health),
        ("Giao thông", traffic),
        ("Hình sự", crime),
        ("Văn hóa", culture),
        ("Giáo dục", education),
        ("Công nghệ", tech),
    ]

    matches = []
    for label, kws in categories:
        if any((kw in joined) or (kw in tokens) for kw in kws):
            matches.append(label)

    if matches:
        label = matches[0]
        if label.startswith("Bóng đá"):
            spec = []
            if "bóng" in tokens or "trận" in tokens or "bóng đá" in joined:
                spec.append("Bóng đá")
            if spec:
                return " ".join([s.capitalize() for s in spec])
        if label == "Kinh tế":
            if any(w in tokens for w in ("giá", "usd", "đồng", "tỷ")):
                return "Kinh tế - Giá cả"
        if label == "Chính trị":
            if "quốc hội" in joined:
                return "Chính trị - Quốc hội"
        if label == "Giao thông":
            if "dự án" in tokens or "cầu" in tokens or "đường" in tokens:
                return "Giao thông - Cơ sở hạ tầng"
        if label == "Sức khỏe":
            if "covid" in tokens or "vắc xin" in joined:
                return "Sức khỏe - Covid-19"
        if label == "Giáo dục":
            if "sinh viên" in tokens or "giảng viên" in tokens:
                return "Giáo dục - Đại học"
        if label == "Văn hóa":
            if "lễ hội" in joined:
                return "Văn hóa - Lễ hội"
        if label == "Công nghệ":
            if "ai" in tokens or "robot" in tokens:
                return "Công nghệ - AI"
        if label == "Thời tiết":
            if "bão" in tokens:
                return "Thời tiết - Bão"
        if label == "Hình sự":
            if "công an" in joined or "khởi tố" in joined:
                return "Hình sự - Pháp luật"
        return label

    top_words = [w.replace("_", " ") for w in keywords[:2]]
    top_words = [w.capitalize() for w in top_words if w]
    if top_words:
        return " - ".join(top_words)
    return "Chủ đề"


def _short_sentence(sent: str, max_len: int = 120) -> str:
    """Truncate sentence if too long (keeps it readable)."""
    if len(sent) <= max_len:
        return sent
    return sent[: max_len - 3].rstrip() + "..."


def generate_topic_description(keywords: List[str], topic_label: str = None, num_keywords: int = 4) -> str:
    """
    Sinh mô tả chủ đề (1-2 câu ngắn gọn, tiếng Việt) dựa trên top keywords.
    - Không sử dụng API ngoài; chỉ heuristics đơn giản.
    - Trả về chuỗi thích hợp cho người đọc tin tức.
    """
    if not keywords:
        return "Mô tả: Không có thông tin."

    readable = [k.replace("_", " ") for k in keywords[:num_keywords]]
    # Build a natural short sentence
    if topic_label and topic_label not in ("Không xác định", ""):
        # Use label and keywords to make a short description
        joined = _join_vietnamese(readable)
        desc = f"Chủ đề {topic_label} - nội dung liên quan đến {joined}."
    else:
        joined = _join_vietnamese(readable)
        desc = f"Bài viết liên quan đến {joined}."

    return _short_sentence(desc)


def main():
    st.title("Demo phân loại chủ đề tin tức — LDA (Tiếng Việt)")
    st.write("Nhập một đoạn văn ngắn bằng tiếng Việt và nhấn **Dự đoán chủ đề**.")

    # Resolve models dir relative to package
    pkg_dir = os.path.dirname(__file__)
    models_dir = os.path.join(pkg_dir, "models")

    try:
        lda_model, count_vectorizer = load_models(models_dir)
    except Exception as e:
        st.error(f"Không thể tải mô hình: {e}")
        return

    text = st.text_area("Văn bản đầu vào", height=200)

    if st.button("Dự đoán chủ đề"):
        if not text or not text.strip():
            st.warning("Vui lòng nhập văn bản để dự đoán.")
            return

        # Tiền xử lý
        processed = preprocess_single_text(text)

        # Vectorize và dự đoán
        doc_vec = count_vectorizer.transform([processed])
        topic_dist = lda_model.transform(doc_vec)[0]
        predicted = int(topic_dist.argmax())

        # Retrieve top keywords for predicted topic
        keywords = get_top_keywords(lda_model, count_vectorizer, predicted, top_n=10)
        keywords_readable = [k.replace("_", " ") for k in keywords]

        # Generate human-readable label and description
        topic_label = generate_topic_title(keywords, top_n=8)
        topic_description = generate_topic_description(keywords, topic_label, num_keywords=4)

        # Generate caption (Đề tài) based on input tokens
        caption = generate_caption_from_input(keywords, processed, num_keywords=5)

        # Display results clearly separated
        st.subheader("Kết quả dự đoán")
        col_id, col_label = st.columns([1, 3])
        col_id.metric("ID chủ đề", str(predicted))
        col_label.markdown(f"**Nhãn chủ đề:** {topic_label}")
        st.markdown(f"**Mô tả chủ đề:** {topic_description}")

        st.subheader("Từ khóa hàng đầu")
        st.write(", ".join(keywords_readable))

        if caption:
            st.markdown(f"**{caption}**")

        st.subheader("Phân phối chủ đề (top 5)")
        topk = sorted(enumerate(topic_dist), key=lambda x: x[1], reverse=True)[:5]
        rows = []
        for tid, prob in topk:
            tid_keywords = get_top_keywords(lda_model, count_vectorizer, tid, top_n=5)
            tid_label = generate_topic_title(tid_keywords, top_n=5)
            tid_desc = generate_topic_description(tid_keywords, tid_label, num_keywords=3)
            rows.append({"Chủ đề (ID)": int(tid), "Nhãn": tid_label, "Mô tả ngắn": tid_desc, "Xác suất": float(round(prob, 4))})

        df = pd.DataFrame(rows)
        st.table(df)


if __name__ == "__main__":
    main()
