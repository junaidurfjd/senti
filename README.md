# Sentiment-Aware Text Classifier

A BERT-based sentiment classifier fine-tuned on the SST-2 dataset for binary sentiment analysis (positive/negative). This project was created as part of the 30 Projects in 30 Days challenge (Day 2).

## Features

- Fine-tuned `bert-base-uncased` on 1,000 training samples from SST-2
- Interactive command-line interface for real-time sentiment analysis
- Logs all predictions with timestamps to `sentiment_log.txt`
- Includes a test suite with 10 diverse test cases
- Reports accuracy and F1-score metrics
- Optimized for 16GB Mac with max sequence length of 512 tokens

## Requirements

- Python 3.8+
- PyTorch 2.2.2
- Transformers
- Datasets
- scikit-learn
- NumPy (<2.0)
- safetensors
- urllib3 (<2.0)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-classifier.git
   cd sentiment-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the classifier:
   ```bash
   python sentiment_classifier.py
   ```

2. The script will:
   - Load and preprocess the SST-2 dataset
   - Fine-tune the BERT model (if not already trained)
   - Run the test suite and display metrics
   - Enter interactive mode for manual input

3. In interactive mode:
   - Enter any text to get sentiment analysis
   - Type 'quit' to exit

## Test Suite Results

The model is evaluated on the following test cases:
- I love coding!
- This movie is awesome!
- It's okay but could be better.
- I'm Jane, this app is great!
- The service was terrible.
- Not bad, but not great either.
- I had a wonderful experience!
- This is the worst product ever.
- It's fine, I guess.
- Absolutely fantastic!

## Logging

All predictions are logged to `sentiment_log.txt` in the format:
```
Timestamp | Input | Sentiment | Confidence
--------------------------------------------------
2023-01-01 12:00:00 | I love this! | positive | 0.9876
```

## Performance

- Training Samples: 1,000
- Test Samples: 200
- Max Sequence Length: 512
- Optimized for: 16GB Mac

## License

MIT
