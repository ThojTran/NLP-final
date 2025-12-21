# Nghiên cứu Chủ đề Tin tức tiếng Việt Nguồn dữ liệu Báo Mới (NLP)

**Mô tả ngắn:**
Dự án thực hiện pipeline xử lý dữ liệu tin tức tiếng Việt: tiền xử lý (cleaning, tokenization), vector hóa (BoW, TF-IDF, Embedding), và trích chủ đề bằng nhiều phương pháp (LDA, LSA, KMeans, HAC, BERTopic). Mục tiêu là tạo ra các kết quả trong thư mục `Output/` và dễ dàng mở rộng.

---

## Tổng quan
- Dữ liệu đầu vào: các file trong `Data/` (ví dụ `news_DB.csv`, `final_data.csv`).
  - File `news_DB.csv` là file crawl data từ nguồn Báo Mới về và tiến hành tiền xử lý dữ liệu (Pre-Processing) cho ra file dữ liệu sạch `final_data.csv`.
  - Dùng `final_data.csv` để train mô hình.
- Notebooks chính:
  - `Pre-Processing.ipynb` — tiền xử lý dữ liệu (lọc, chuẩn hóa, tokenize, loại stopwords, xử lý viết tắt).
  - `Vectorize and Model/vectorize_BoW.ipynb` — vector hóa BoW + LDA/LSA/KMeans/HAC.
  - `Vectorize and Model/vectorize_TFIDF.ipynb` — TF-IDF + các mô hình chủ đề/cluster.
  - `Vectorize and Model/vectorize_Embedding.ipynb` — embedding (sentence-transformers) + clustering.
  - `Vectorize and Model/BerTopic.ipynb` — chạy BERTopic để lấy chủ đề và biểu diễn.
- Kết quả lưu trong `Output/` theo từng phương pháp (CSV, HTML visualizations, hình ảnh):
  - `Output/BoW/` — ví dụ: `final_data_with_all_topics.csv`
  - `Output/TF-IDF results/` — visualizations, comparison charts
  - `Output/Embedding/` — kết quả embedding-based
  - `Output/BERtopic/` — `topic_captions_final.csv`, các summary

---

## Hướng chạy (recommended order)
1. Chạy `Pre-Processing.ipynb` để chuẩn hóa dữ liệu (sinh `Data/final_data.csv` hoặc tương tự).
2. Chạy các notebook vectorize theo nhu cầu:
   - `vectorize_BoW.ipynb` → tạo `Output/BoW/*`.
   - `vectorize_TFIDF.ipynb` → tạo `Output/TF-IDF results/*`.
   - `vectorize_Embedding.ipynb` → tạo `Output/Embedding/*`.
3. Chạy `BerTopic.ipynb` để lấy kết quả BERTopic (ví dụ sau khi chạy `Output/BERtopic/topic_captions_final. csv`).
4. Mở các file `.html` trong `Output/` bằng trình duyệt để xem trực quan (ví dụ `lda_visualization.html`).

---

## Yêu cầu môi trường
- Python 3.8+ (3.9/3.10 đề xuất).
- Tạo virtualenv (Windows):
  - python -m venv .venv
  - .venv\Scripts\activate
- Cài các package cơ bản (gợi ý):
  - pip install -U pip
  - pip install pandas numpy scikit-learn gensim matplotlib seaborn notebook jupyterlab bertopic sentence-transformers umap-learn hdbscan plotly
- Lưu ý: một số notebook sử dụng thêm thư viện khác — kiểm tra cell `import` trong từng notebook để chắc chắn.

---

## Ghi chú và format đầu ra
- `Output/BoW/final_data_with_all_topics.csv` có các cột như: `description, category, topic_LDA, topic_prob_LDA, topic_LSA, topic_prob_LSA, ...` — dùng để so sánh kết quả giữa phương pháp.
- `Output/BERtopic/topic_captions_final.csv` bao gồm: `Topic, Count, Ten_Chu_De, Representation` — thuận tiện cho báo cáo tóm tắt chủ đề.

---

