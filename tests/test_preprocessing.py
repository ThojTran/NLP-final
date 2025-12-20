import os
import tempfile
import shutil
import pandas as pd

from topic_modeling_app.preprocessing.preprocessing import VietnameseTextPreprocessor, preprocess_single_text


def _write_temp_file(content: str, suffix: str = '') -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


def test_acronyms_expand_variants():
    acr_csv = "acronym_words,fullform\nTP.HCM,Thành phố Hồ Chí Minh\nTP-HCM,Thành phố Hồ Chí Minh\n"
    acr_path = _write_temp_file(acr_csv, suffix='.csv')
    sw_path = _write_temp_file("và\nlà\ncủa\n", suffix='.txt')

    proc = VietnameseTextPreprocessor(sw_path, acr_path)

    assert "Thành phố Hồ Chí Minh" in proc.expand_acronyms("TP.HCM")
    assert "Thành phố Hồ Chí Minh" in proc.expand_acronyms("tp.hcm")
    assert "Thành phố Hồ Chí Minh" in proc.expand_acronyms("TP-HCM")

    os.remove(acr_path)
    os.remove(sw_path)


def test_clean_text_removes_url_emoji_digits():
    sw_path = _write_temp_file("", suffix='.txt')
    acr_path = _write_temp_file("", suffix='.csv')
    proc = VietnameseTextPreprocessor(sw_path, acr_path)

    text = "Check http://example.com 😊 số 12345"
    cleaned = proc.clean_text(text)
    assert "http" not in cleaned
    assert "😊" not in cleaned
    assert "12345" not in cleaned

    os.remove(acr_path)
    os.remove(sw_path)


def test_preprocess_single_text_with_cache_and_instance():
    acr_csv = "acronym_words,fullform\nTP.HCM,Thành phố Hồ Chí Minh\n"
    acr_path = _write_temp_file(acr_csv, suffix='.csv')
    sw_path = _write_temp_file("và\nlà\n", suffix='.txt')

    # test using cached factory via paths
    out = preprocess_single_text("TP.HCM 123", stopwords_path=sw_path, acronyms_path=acr_path)
    assert "Thành phố" in out

    # test using an existing instance to avoid reloading files
    preprocessor = VietnameseTextPreprocessor(sw_path, acr_path)
    out2 = preprocess_single_text("TP.HCM 456", preprocessor=preprocessor)
    assert "Thành phố" in out2

    os.remove(acr_path)
    os.remove(sw_path)