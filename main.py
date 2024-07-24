"""
Main entry point for the KIE system
"""
from transformers import AutoTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import config
from src.data_loaders.data_loader import load_data, OpenFoodFactsDataset
from src.processing.text_processing import preprocess_data, normalize_text, tokenize
from src.entity_extraction.ml_tasks import single_label_classification, multi_label_classification, entity_tagging
from src.entity_extraction.models import SimpleClassifier, train, evaluate

def main():
    # Load and preprocess data
    raw_data = load_data(config.DATA_PATH, nrows=config.MAX_DATA_SUBSAMPLE)
    train_data, eval_data = preprocess_data(raw_data, 
                                            test_size=config.VALIDATION_SPLIT,
                                            random_state=config.RANDOM_SEED)
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_MODEL)

    # Create PyTorch Datasets
    train_dataset = OpenFoodFactsDataset(train_data['text'].tolist(), 
                                         train_data['labels'].tolist(), 
                                         tokenizer,
                                        config.MAX_LENGTH)
    
    eval_dataset = OpenFoodFactsDataset(eval_data['text'].tolist(),
                                         eval_data['labels'].tolist(), 
                                         tokenizer, 
                                         config.MAX_LENGTH)
    # Load data using Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    if config.WORKFLOW == 'TRAIN':
        num_classes = 10  # Sample number of classes for classification
        
        # Define model, optimizer, and loss function
        model = SimpleClassifier(input_size=config.MAX_LENGTH, 
                                hidden_size=config.HIDDEN_SIZE, 
                                num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Device configuration (CPU or GPU)
        model.to(config.DEVICE)

        # Example of using the dataloaders for training or evaluation
        num_epochs = 10
        for epoch in range(num_epochs):
            train_loss = train(model, train_dataloader, optimizer, criterion, config.DEVICE)
            eval_loss = evaluate(model, eval_dataloader, criterion, config.DEVICE)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}')

    elif config.WORKFLOW == 'INFERENCE':
        # Choose Text Pre-Processing
        sample_text = "Sample product description"
#        normalized_text = normalize_text(sample_text)
#        tokens = tokenizer.encode (normalized_text)

        # Select ML Workflow
        single_label_classification(features, labels)
#        multi_label_classification(features, multi_labels)
#        tagged_entities = entity_tagging(sample_text)


if __name__ == "__main__":
    main()