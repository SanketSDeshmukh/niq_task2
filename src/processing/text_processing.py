"""
Module for text normalization and processing
"""
from typing import List
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config 

def normalize_text(text: str) -> str:
    """
    Normalize the input text
    :param text: Input text to normalize
    :return: Normalized text
    """
    normalized_text = text.lower()
    return normalized_text

def preprocess_data(data: pd.DataFrame, test_size=.2, random_state=42) -> pd.DataFrame:
    """
    Preprocess the loaded data
    :param data: Raw data as a Pandas DataFrame
    :test_size: Split Size for Dataset
    :random_state: Random State
    :return: Preprocessed data as a Pandas DataFrame
    """

    data['text'] = data['text'].apply(normalize_text)

    # Split data into training and evaluation sets
    train_data, eval_data = train_test_split(data,
                                             test_size=test_size, 
                                             random_state=random_state)

    return train_data, eval_data
