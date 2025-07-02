from transformers import BertTokenizer, BertForSequenceClassification
import torch

def load_model():
    """Load pre-trained BERT model and tokenizer for text classification."""
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for a given text using the BERT model."""
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process output
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    
    return "positive" if predicted_class == 1 else "negative", confidence

def main():
    # Load model and tokenizer
    print("Loading BERT model and tokenizer...")
    model, tokenizer = load_model()
    
    # Example texts
    texts = [
        "I love this movie! It's fantastic!",
        "This is the worst product I've ever bought.",
        "The weather is nice today.",
        "I'm not sure how I feel about this."
    ]
    
    # Make predictions
    print("\nSentiment Analysis Results:")
    print("-" * 50)
    for text in texts:
        sentiment, confidence = predict_sentiment(text, model, tokenizer)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
        print("-" * 50)

if __name__ == "__main__":
    main()
