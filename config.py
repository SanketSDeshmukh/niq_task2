"""
Configuration file to store all parameters, file paths, and settings
"""
DATA_PATH = "data/en.openfoodfacts.org.products.csv"
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42
MAX_DATA_SUBSAMPLE = 99999
TOKENIZER_MODEL = 'bert-base-cased'
MAX_LENGTH = 128
HIDDEN_SIZE = 64
DEVICE = 'cpu'
WORKFLOW = 'TRAIN'
