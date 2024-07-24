import pytest 
from src.processing.text_processing import normalize_text

def test_normalize_text():
    assert normalize_text('ABCD') == 'abcd'
