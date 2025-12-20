import os
import pandas as pd

# Import using package-relative import first (works when running as a module).
# Fall back to absolute package import for other execution styles.
try:
    from ..preprocessing.preprocessing import preprocess_dataframe
except Exception:
    from topic_modeling_app.preprocessing.preprocessing import preprocess_dataframe

# Resolve data paths relative to package root (robust for different working dirs)
_pkg_dir = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_PATH = os.path.join(_pkg_dir, "data", "news_DB.csv")
STOPWORDS_PATH = os.path.join(_pkg_dir, "data", "vietnamese-stopwords-dash.txt")
ACRONYMS_PATH = os.path.join(_pkg_dir, "data", "viet_tat_clean.txt")
OUTPUT_PATH = os.path.join(_pkg_dir, "data", "processed_data.csv")

def main():
    print("🚀 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")

    print("🚀 Running preprocessing...")
    processed_df = preprocess_dataframe(
        df,
        stopwords_path=STOPWORDS_PATH,
        acronyms_path=ACRONYMS_PATH,
        min_words=20
    )

    print("💾 Saving processed data...")
    processed_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"✅ Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()