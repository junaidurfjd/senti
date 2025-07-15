import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentClassifier:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)
        self.max_length = max_length
        self.id2label = {0: "negative", 1: "positive"}
        
        # Create log file if it doesn't exist
        if not os.path.exists('sentiment_log.txt'):
            with open('sentiment_log.txt', 'w') as f:
                f.write("Timestamp | Input | Sentiment | Confidence\n")
                f.write("-" * 80 + "\n")

    def preprocess_function(self, examples):
        return self.tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def load_dataset(self, dataset_name='glue', config_name='sst2', train_samples=1000, test_samples=200):
        logger.info("Loading SST-2 dataset...")
        dataset = load_dataset(dataset_name, config_name)
        
        # Select subset of data
        train_dataset = dataset['train'].select(range(train_samples))
        test_dataset = dataset['validation'].select(range(test_samples))
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['sentence'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            
        train_encodings = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['sentence', 'idx']
        )
        
        test_encodings = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['sentence', 'idx']
        )
        
        train_encodings.set_format(type='torch', 
                                 columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
        test_encodings.set_format(type='torch',
                                 columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
        
        return train_encodings, test_encodings

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {'accuracy': acc, 'f1': f1}

    def train(self, train_dataset, test_dataset, output_dir='./results', num_train_epochs=3, batch_size=8):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )

        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Evaluating on test set...")
        eval_results = trainer.evaluate()
        logger.info(f"Test results: {eval_results}")
        
        return trainer

    def predict_sentiment(self, text):
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()
            
        sentiment = self.id2label[pred_label]
        
        self.log_prediction(text, sentiment, confidence)
        
        return sentiment, confidence
    
    def log_prediction(self, text, sentiment, confidence):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | {text} | {sentiment} | {confidence:.4f}\n"
        
        with open('sentiment_log.txt', 'a') as f:
            f.write(log_entry)
        
        logger.info(f"Input: {text}")
        logger.info(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")

def run_test_suite(classifier, test_cases):
    logger.info("\n" + "="*50)
    logger.info("RUNNING TEST SUITE")
    logger.info("="*50)
    
    true_labels = []
    pred_labels = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTest Case {i}/{len(test_cases)}:")
        sentiment, confidence = classifier.predict_sentiment(test_case)
        pred_labels.append(1 if sentiment == 'positive' else 0)
        # For demo purposes, we'll assume all test cases are positive
        # In a real scenario, you would have true labels for evaluation
        true_labels.append(1)  # This is just a placeholder
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    logger.info("\n" + "="*50)
    logger.info(f"TEST SUITE RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info("="*50 + "\n")
    
    return accuracy, f1

def main():
    # Test cases
    test_cases = [
        'I love coding!',
        'This movie is awesome!',
        'It\'s okay but could be better.',
        'I\'m Jane, this app is great!',
        'The service was terrible.',
        'Not bad, but not great either.',
        'I had a wonderful experience!',
        'This is the worst product ever.',
        'It\'s fine, I guess.',
        'Absolutely fantastic!'
    ]
    
    # Initialize classifier
    classifier = SentimentClassifier(max_length=512)
    
    # Load and prepare dataset
    train_dataset, test_dataset = classifier.load_dataset()
    
    # Train the model
    trainer = classifier.train(train_dataset, test_dataset)
    
    # Run test suite
    accuracy, f1 = run_test_suite(classifier, test_cases)
    
    # Interactive mode
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS INTERACTIVE MODE")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        user_input = input("Enter text to analyze (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        sentiment, confidence = classifier.predict_sentiment(user_input)
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    main()
