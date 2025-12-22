import streamlit as st
import pandas as pd
from bertopic import BERTopic
import os
from sentence_transformers import SentenceTransformer
import re

# Import underthesea cho tiền xử lý tiếng Việt
try:
    from underthesea import word_tokenize, sent_tokenize
except ImportError:
    st.error("Vui lòng cài đặt underthesea: pip install underthesea")
    st.stop()

# --- CẤU HÌNH ĐƯỜNG DẪN ---
TOPIC_INFO_PATH = r'C:\Users\PC\Desktop\BERTopic Model\reports\figures\topic_captions_final.csv'
MODEL_PATH = r'C:\Users\PC\Desktop\BERTopic Model\models\bertopic_model_colab'
EMBEDDING_MODEL_NAME = 'keepitreal/vietnamese-sbert'

# --- WRAPPER CLASS CHO EMBEDDING MODEL ---
class EmbeddingModelWrapper:
    """Wrapper để BERTopic có thể sử dụng SentenceTransformer"""
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts, verbose=False, **kwargs):
        """Phương thức mà BERTopic cần - chấp nhận mọi tham số"""
        return self.model.encode(texts, show_progress_bar=verbose)
    
    def embed(self, texts, verbose=False, **kwargs):
        """Phương thức dự phòng"""
        return self.embed_documents(texts, verbose=verbose)

# --- HÀM TẢI TÀI NGUYÊN ---
@st.cache_resource
def load_resources():
    """Tải mô hình BERTopic và dữ liệu topic captions"""
    
    # Tải DataFrame chứa thông tin các chủ đề
    if os.path.exists(TOPIC_INFO_PATH):
        df_topics = pd.read_csv(TOPIC_INFO_PATH)
    else:
        st.error(f"❌ Không tìm thấy file: {TOPIC_INFO_PATH}")
        df_topics = pd.DataFrame()

    # Tải mô hình BERTopic
    if os.path.exists(MODEL_PATH):
        try:
            model = BERTopic.load(MODEL_PATH)
            
            # Sử dụng wrapper class cho embedding model
            embedding_model = EmbeddingModelWrapper(EMBEDDING_MODEL_NAME)
            model.embedding_model = embedding_model
            
            st.success("✅ Mô hình BERTopic đã được tải thành công!")
            
        except Exception as e:
            st.error(f"❌ Lỗi khi tải mô hình: {e}")
            model = None
    else:
        st.error(f"❌ Không tìm thấy mô hình tại: {MODEL_PATH}")
        model = None
        
    return df_topics, model

# --- HÀM TÓM TẮT VÃN BẢN ---
def summarize_text(text, num_sentences=3):
    """
    Tóm tắt văn bản bằng cách lấy N câu đầu tiên
    (Có thể thay thế bằng các thuật toán phức tạp hơn)
    """
    try:
        # Tách thành các câu
        sentences = sent_tokenize(text)
        
        # Lấy số câu cần thiết
        if len(sentences) <= num_sentences:
            return text
        else:
            summary = ' '.join(sentences[:num_sentences])
            return summary + "..."
            
    except Exception as e:
        # Fallback: Lấy 300 ký tự đầu
        return text[:300] + "..." if len(text) > 300 else text

# --- HÀM TIỀN XỬ LÝ VĂN BẢN ---
def preprocess_text(text):
    """
    Tiền xử lý văn bản tiếng Việt:
    - Làm sạch văn bản
    - Tách từ (tokenization)
    """
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tách từ tiếng Việt
    tokenized = word_tokenize(text, format="text")
    
    return tokenized

# --- HÀM DỰ ĐOÁN TOPIC ---
def predict_topic(model, text, df_topics):
    """
    Dự đoán topic ID và lấy thông tin chi tiết
    """
    try:
        # Tạo embedding trước
        embeddings = model.embedding_model.model.encode([text])
        
        # Dự đoán topic với embedding đã tạo sẵn
        topics, probs = model.transform([text], embeddings)
        topic_id = topics[0]
        confidence = probs[0][topic_id] if probs is not None else None
        
        # Tìm thông tin topic trong DataFrame
        topic_row = df_topics[df_topics['Topic'] == topic_id]
        
        if not topic_row.empty:
            topic_name = topic_row.iloc[0]['Ten_Chu_De']
            keywords = topic_row.iloc[0]['Representation']
        else:
            topic_name = "Không xác định"
            keywords = "Không có từ khóa"
            
        return {
            'topic_id': topic_id,
            'topic_name': topic_name,
            'keywords': keywords,
            'confidence': confidence
        }
        
    except Exception as e:
        st.error(f"Lỗi khi dự đoán topic: {e}")
        return None

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="BERTopic News Classifier", 
    page_icon="📰", 
    layout="wide"
)

# --- HEADER ---
st.title("📰 Hệ thống Phân loại & Tóm tắt Tin tức")
st.markdown("""
**Chức năng:**
- 🔍 Phân loại văn bản vào các chủ đề tin tức
- 📝 Tóm tắt nội dung chính
- 🏷️ Hiển thị từ khóa đặc trưng

*Powered by BERTopic & Vietnamese SBERT*
""")

# Tải tài nguyên
df_topics, bert_model = load_resources()

st.markdown("---")

# --- GIAO DIỆN NHẬP LIỆU ---
col_input, col_settings = st.columns([3, 1])

with col_input:
    user_input = st.text_area(
        "📄 Nhập văn bản tin tức (tiếng Việt):",
        height=200,
        placeholder="Ví dụ: Ngân hàng Nhà nước vừa công bố số liệu về lãi suất và tỷ giá trong quý 3 năm 2024..."
    )

with col_settings:
    st.markdown("### ⚙️ Cài đặt")
    num_summary_sentences = st.slider(
        "Số câu tóm tắt:", 
        min_value=1, 
        max_value=5, 
        value=3
    )
    
    show_tokenized = st.checkbox("Hiển thị văn bản sau tách từ", value=False)

# --- NÚT THỰC HIỆN ---
if st.button("🚀 Phân tích văn bản", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Vui lòng nhập văn bản để phân tích!")
    elif bert_model is None:
        st.error("❌ Mô hình chưa được tải thành công. Vui lòng kiểm tra đường dẫn.")
    else:
        with st.spinner('🔄 Đang xử lý văn bản...'):
            
            # BƯỚC 1: Tóm tắt văn bản
            summary_text = summarize_text(user_input, num_summary_sentences)
            
            # BƯỚC 2: Tiền xử lý văn bản
            processed_text = preprocess_text(user_input)
            
            # BƯỚC 3: Dự đoán Topic
            result = predict_topic(bert_model, processed_text, df_topics)
            
            if result:
                # --- HIỂN THỊ KẾT QUẢ ---
                st.success("✅ Phân tích hoàn tất!")
                st.markdown("---")
                
                # Layout 2 cột
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🎯 Kết quả Phân loại")
                    
                    # Topic ID
                    st.metric(label="Topic ID", value=result['topic_id'])
                    
                    # Tên chủ đề
                    st.markdown(f"**📌 Chủ đề:** `{result['topic_name']}`")
                    
                    # Độ tin cậy (nếu có)
                    if result['confidence'] is not None:
                        confidence_percent = result['confidence'] * 100
                        st.progress(result['confidence'])
                        st.caption(f"Độ tin cậy: {confidence_percent:.2f}%")
                    
                    # Từ khóa đặc trưng
                    st.markdown("**🔑 Từ khóa đặc trưng:**")
                    st.code(result['keywords'], language="python")
                    
                    # Văn bản sau tách từ (tùy chọn)
                    if show_tokenized:
                        with st.expander("👀 Xem văn bản sau tách từ"):
                            st.text(processed_text)
                
                with col2:
                    st.subheader("📋 Tóm tắt Nội dung")
                    st.info(summary_text)
                    
                    # Thống kê văn bản
                    st.markdown("**📊 Thống kê:**")
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        st.metric("Độ dài gốc", f"{len(user_input)} ký tự")
                    with stats_col2:
                        st.metric("Độ dài tóm tắt", f"{len(summary_text)} ký tự")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/4A90E2/FFFFFF?text=BERTopic", use_container_width=True)
    
    st.markdown("## 📖 Hướng dẫn sử dụng")
    st.markdown("""
    1. Dán văn bản tin tức vào ô nhập liệu
    2. Chọn số câu muốn tóm tắt
    3. Nhấn nút **Phân tích văn bản**
    4. Xem kết quả phân loại và tóm tắt
    """)
    
    st.markdown("---")
    st.markdown("## ℹ️ Thông tin Hệ thống")
    st.info(f"""
    **Mô hình:** BERTopic  
    **Embedding:** {EMBEDDING_MODEL_NAME}  
    **Tiền xử lý:** Underthesea  
    **Số chủ đề:** {len(df_topics) if not df_topics.empty else 'N/A'}
    """)
    
    # Hiển thị danh sách các topic (tùy chọn)
    if not df_topics.empty and st.checkbox("Xem danh sách chủ đề"):
        st.dataframe(
            df_topics[['Topic', 'Ten_Chu_De']].head(10),
            use_container_width=True,
            hide_index=True
        )

# --- FOOTER ---
st.markdown("---")
st.caption("© 2024 BERTopic News Classifier | Developed with Streamlit")