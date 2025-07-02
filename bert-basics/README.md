# BERT Basics Playground

This directory contains example scripts to explore different capabilities of BERT without any fine-tuning. Each script demonstrates a different aspect of BERT's functionality using pre-trained models from Hugging Face's Transformers library.

## Scripts Overview

1. **`bert_text_classification.py`**
   - Basic text classification using BERT
   - Classifies text as positive or negative sentiment
   - No fine-tuning, uses pre-trained BERT as-is

2. **`bert_question_answering.py`**
   - Question answering using BERT fine-tuned on SQuAD
   - Takes a context paragraph and answers questions about it
   - Demonstrates BERT's ability to understand and extract information

3. **`bert_token_analysis.py`**
   - Explores BERT's tokenization and embeddings
   - Shows word similarities using BERT's embeddings
   - Visualizes self-attention weights

4. **`bert_text_generation.py`**
   - Text generation using BERT
   - Shows how BERT can be used for creative text generation
   - Includes temperature sampling for diverse outputs

## Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run any of the scripts:
   ```bash
   python bert_text_classification.py
   python bert_question_answering.py
   python bert_token_analysis.py
   python bert_text_generation.py
   ```

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- NumPy
- Matplotlib
- scikit-learn
- tqdm

## Notes

- All scripts use pre-trained models and don't require GPU (though it will be faster with one)
- The models will be downloaded automatically on first run
- No fine-tuning is performed - these are out-of-the-box examples

## Expected Outputs

- `bert_attention.png`: Visualization of BERT's self-attention weights
- Console output showing model predictions and analysis

## Next Steps

1. Experiment with different pre-trained BERT models
2. Try different input texts and questions
3. Explore the attention mechanisms in more detail
4. Use these examples as a starting point for your own BERT-based projects
