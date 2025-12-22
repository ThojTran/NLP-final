# ỨNG DỤNG TOPIC MODELLING LDA VÀO DỮ LIỆU THU THẬP TỪ BÁO MỚI

## Giới thiệu
**Topic Modeling App** là một project học thuật minh họa quy trình tiền xử lý văn bản tiếng Việt và mô hình hóa chủ đề bằng **Latent Dirichlet Allocation (LDA)**. Mục tiêu: mã nguồn đơn giản, trực quan, dễ giải thích cho báo cáo môn học.

---

## Cài đặt môi trường & dependencies

1. Tạo môi trường ảo (Windows PowerShell ví dụ):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Cài dependencies:
```bash
pip install -r requirements.txt
```

Hoặc sẽ cài riêng Streamlit: `pip install streamlit`.

---

## Cách chạy

> **Luôn chạy từ thư mục gốc của repository**  và **dùng `python -m ...` để chạy modules**. Không chạy các file `.py` trực tiếp (ví dụ `python data/run_preprocessing.py`) — điều này có thể gây lỗi import.  
> Nếu vẫn không chạy được. Hãy mở powershell chạy đúng thư mục gốc xong nhập  $env:PYTHONPATH = (Get-Location).Path; streamlit run .\topic_modeling_app\app.py

Ví dụ các lệnh (từ thư mục gốc):

- Chạy tiền xử lý dữ liệu:
```bash
python -m data.run_preprocessing
```

- Huấn luyện LDA:
```bash
python -m models.train_lda
```

- Kiểm tra hàm dự đoán (module demo):
```bash
python -m inference.predict_topic
```

- Chạy giao diện Streamlit (UI):
```bash
streamlit run topic_modeling_app/app.py
```

- Chạy test:
```bash
pytest
```

>  Nếu gặp `ModuleNotFoundError`, đảm bảo bạn đang ở **thư mục gốc** và chạy bằng `python -m ...`. Bạn cũng có thể thiết lập `PYTHONPATH` tạm thời tới thư mục gốc nếu cần.

---

## Cấu trúc thư mục (tổng quan)

```
NLP-ung dung/
├─ app.py
├─ data/
│  └─ run_preprocessing.py
├─ models/
│  └─ train_lda.py
├─ preprocessing/
│  └─ preprocessing.py
├─ inference/
│  ├─ predict_topic.py
│  └─ recommend.py
├─ utils/
│  ├─ loader.py
│  └─ similarity.py
├─ models/  # lưu lda_model.pkl, count_vectorizer.pkl sau khi train
└─ requirements.txt
```

---

## Cách import module

Project được thiết kế như một package Python; **hãy dùng import tuyệt đối (absolute import)** để mã chạy ổn định khi dùng `python -m` và khi cài package:

```python
from preprocessing.preprocessing import preprocess_single_text
from utils.loader import load_models
```

---

## Ghi chú về `__init__.py` & packaging

- Giữ file `__init__.py` trong các thư mục package (ví dụ `preprocessing`, `models`, `utils`) để Python nhận diện package.
- Nếu cần phân phối / cài đặt, có thể thêm `pyproject.toml` / `setup.cfg` sau này — không bắt buộc cho đồ án.

---

## Troubleshooting nhanh

- `ModuleNotFoundError`: kiểm tra bạn đang ở thư mục gốc chưa và có chạy bằng `python -m` không.
- Thiếu dữ liệu: chạy `python -m data.run_preprocessing` trước để tạo `processed_data.csv`.
- Thiếu model `.pkl`: chạy `python -m models.train_lda`.

---


