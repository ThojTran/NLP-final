import re
import string
import pandas as pd
from underthesea import word_tokenize, sent_tokenize


class VietnameseTextPreprocessor:
    """
    Thanh đơn giản cho tiền xử lý văn bản tiếng Việt.
    Sử dụng cho pipeline: lower -> expand acronyms -> clean -> tokenize -> remove stopwords -> filter tokens

    Improvements added (still simple):
    - support for a small built-in stopword set when no file is provided
    - ability to pass extra stopwords
    - stronger token filtering to remove non-letter tokens
    """

    # A small, useful set of common Vietnamese stopwords (kept intentionally short)
    DEFAULT_STOPWORDS = {
        "và", "là", "của", "có", "cho", "đã", "nhưng", "nếu", "trong",
        "với", "một", "từ", "theo", "do", "sau", "khi", "để", "vì",
    }

    def __init__(self, stopwords_path: str = None, acronyms_path: str = None, extra_stopwords=None):
        """Initialize preprocessor.

        Args:
            stopwords_path: path to a newline-separated stopwords file
            acronyms_path: path to a CSV mapping acronyms to full form
            extra_stopwords: optional iterable of additional stopwords
        """
        file_sw = self._load_stopwords(stopwords_path)
        self.stopwords = set().union(self.DEFAULT_STOPWORDS, file_sw)
        if extra_stopwords:
            self.stopwords.update({s.lower() for s in extra_stopwords})

        self.acronyms = self._load_acronyms(acronyms_path)

    def _load_stopwords(self, path: str):
        """Load stopwords từ file text (mỗi dòng 1 từ). Trả về set."""
        if not path:
            return set()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return set(line.strip().lower() for line in f if line.strip())
        except Exception:
            return set()

    def _load_acronyms(self, path: str):
        """Load từ viết tắt từ CSV đơn giản (cột 'acronym_words', 'fullform')."""
        if not path:
            return {}
        try:
            df = pd.read_csv(path, encoding='utf-8')
            if 'acronym_words' not in df.columns or 'fullform' not in df.columns:
                return {}
            acr = {}
            for k, v in zip(df['acronym_words'], df['fullform']):
                if pd.isna(k) or pd.isna(v):
                    continue
                key = str(k).strip().lower()
                acr[key] = str(v).strip()
            return acr
        except Exception:
            return {}

    def expand_acronyms(self, text: str) -> str:
        """Thay các từ viết tắt bằng dạng đầy đủ nếu có trong dictionary (simple lookup)."""
        words = text.split()
        out = []
        for w in words:
            key = w.strip(string.punctuation).lower()
            if key in self.acronyms:
                out.append(self.acronyms[key])
            else:
                out.append(w)
        return ' '.join(out)

    def clean_text(self, text: str) -> str:
        """Làm sạch văn bản: bỏ URL/email/số, giữ ký tự chữ và dấu cách, loại punctuation và emoji (đơn giản)."""
        if not isinstance(text, str):
            return ''
        # remove urls and emails
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        # remove digits
        text = re.sub(r'\d+', ' ', text)
        # keep letters (latin + Vietnamese range) and spaces; replace others with space
        text = re.sub(r'[^a-zA-ZÀ-ỹ\s]', ' ', text)
        # collapse spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_vietnamese(self, text: str) -> str:
        """Tokenize using underthesea. Return a single string of tokens separated by spaces."""
        try:
            sentences = sent_tokenize(text)
            tokenized = [word_tokenize(s, format='text') for s in sentences]
            return ' '.join(tokenized)
        except Exception:
            # fallback: return original text (already cleaned)
            return text

    def remove_stopwords(self, text: str) -> str:
        """Loại bỏ stopwords (lowercase match). Strip punctuation when checking."""
        words = text.split()
        kept = []
        for w in words:
            key = w.strip(string.punctuation).lower()
            if key and key not in self.stopwords:
                kept.append(w)
        return ' '.join(kept)

    def remove_invalid_words(self, text: str, min_len: int = 2, max_len: int = 30) -> str:
        """Loại bỏ từ quá ngắn/quá dài, token vô nghĩa, hoặc không chứa ký tự chữ."""
        words = text.split()
        valid = []
        for w in words:
            if len(w) < min_len or len(w) > max_len:
                continue
            if len(set(w)) == 1:  # all same character like '!!!!!' or 'aaaaa'
                continue
            # Require at least one alphabetic character (latin or Vietnamese range)
            if not re.search(r"[a-zA-ZÀ-ỹ]", w):
                continue
            valid.append(w)
        return ' '.join(valid)

    def preprocess_text(self, text: str) -> str:
        """
        Một pipeline đơn giản cho 1 đoạn văn:
        - lower
        - expand acronyms
        - clean (remove urls, digits, punctuation)
        - tokenize (underthesea)
        - remove stopwords
        - remove invalid words
        """
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = self.expand_acronyms(text)
        text = self.clean_text(text)
        text = self.tokenize_vietnamese(text)
        text = self.remove_stopwords(text)
        text = self.remove_invalid_words(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def preprocess_dataframe(df: pd.DataFrame, stopwords_path: str = None, acronyms_path: str = None, min_words: int = 20, extra_stopwords=None) -> pd.DataFrame:
    """
    Tiền xử lý toàn bộ DataFrame cho training.
    Yêu cầu df có cột: 'title', 'description', 'category' (category có thể là None).
    Trả về DataFrame với các cột: title, description, title_processed, description_processed, combined_text, category
    """
    preprocessor = VietnameseTextPreprocessor(stopwords_path, acronyms_path, extra_stopwords)

    # drop rows without title/description
    df = df.copy()
    df = df.dropna(subset=['title', 'description'])
    df = df[df['title'].str.strip() != '']
    df = df[df['description'].str.strip() != '']

    df['title_processed'] = df['title'].apply(preprocessor.preprocess_text)
    df['description_processed'] = df['description'].apply(preprocessor.preprocess_text)
    df['combined_text'] = (df['title_processed'] + ' ' + df['description_processed']).str.strip()
    df['word_count'] = df['combined_text'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= min_words]

    result = df[['title', 'description', 'title_processed', 'description_processed', 'combined_text', 'category']].reset_index(drop=True)
    return result


def preprocess_single_text(text: str, stopwords_path: str = None, acronyms_path: str = None, preprocessor: VietnameseTextPreprocessor = None) -> str:
    """Tiền xử lý 1 đoạn văn cho inference.

    If a `preprocessor` instance is provided it is reused; otherwise a new
    one is created using the given paths. This avoids reloading stopwords on
    every call when used in a loop or a server.
    """
    if preprocessor is None:
        preprocessor = VietnameseTextPreprocessor(stopwords_path, acronyms_path)
    return preprocessor.preprocess_text(text)