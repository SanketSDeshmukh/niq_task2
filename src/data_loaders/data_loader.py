"""
Module for loading and preprocessing the data
"""
from typing import Tuple, List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import config
from src.processing.text_processing import normalize_text

class OpenFoodFactsDataset(Dataset):
    def __init__(self, 
                 data: pd.DataFrame,
                 tokenizer, 
                 targets,
                 max_length,
                 transform=None
                 ):
        """
        Initialize the dataset
        :param data: Pandas DataFrame containing the Open Food Facts data
        :param transform: Optional transform to be applied on a sample
        """
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_length = max_length

    def __len__(self) -> int:
        """
        :return: The number of samples in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        :param idx: Index of the item to retrieve
        :return: A dict containing the features and labels for the given index
        """
        try:
            encoded_input = self.tokenizer.encode_plus(
                self.data[idx],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded_input['input_ids'].flatten(),
                'attention_mask': encoded_input['attention_mask'].flatten(),
                'targets': torch.tensor(self.targets[idx], dtype=torch.long)
            }
        except Exception as err:
            raise Exception(str(err)) from err
        
def load_data( file_path: str, sep='\t',nrows=None ) -> pd.DataFrame:
    """
    Load the data from CSV file
    :param file_path: Path to the CSV file
    :return: Pandas DataFrame containing the loaded data
    """
    try:
        print('Data Loading in Progress')
        if nrows:
            df = pd.read_csv(file_path,sep=sep,nrows=nrows)
        else:
            df = pd.read_csv(file_path,sep=sep)
        print(f'Data Loading Completed : {df.shape()}')
        return df
    except Exception as err:
        raise Exception(str(err))