from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def explore_tokenization():
    """Demonstrate BERT tokenization."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    text = "Let's explore BERT tokenization!"
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    print("\nTokenization Example:")
    print("-" * 80)
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Show special tokens
    print("\nSpecial Tokens:")
    print(f"[CLS] token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"[SEP] token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"[PAD] token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

def explore_embeddings():
    """Explore BERT word embeddings."""
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Example words to compare
    words = ["king", "queen", "man", "woman", "computer", "laptop"]
    
    # Get embeddings for each word
    embeddings = {}
    for word in words:
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token embedding as word representation
        embeddings[word] = outputs.last_hidden_state[0, 0, :].numpy()
    
    # Calculate cosine similarity between words
    print("\nWord Similarities:")
    print("-" * 40)
    for i, word1 in enumerate(words):
        for word2 in words[i+1:]:
            sim = cosine_similarity(
                embeddings[word1].reshape(1, -1),
                embeddings[word2].reshape(1, -1)
            )[0][0]
            print(f"{word1} <-> {word2}: {sim:.3f}")

def visualize_attention():
    """Visualize BERT's self-attention (simplified example)."""
    from transformers import BertModel, BertTokenizer
    import torch
    
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    
    # Example sentence
    text = "The cat sat on the mat"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get attention from the first layer and head
    attention = outputs.attentions[0][0, 0].numpy()
    
    # Plot attention weights
    plt.figure(figsize=(10, 8))
    plt.imshow(attention, cmap='viridis')
    plt.xticks(range(len(inputs.input_ids[0])), tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))
    plt.yticks(range(len(inputs.input_ids[0])), tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))
    plt.colorbar()
    plt.title("BERT Self-Attention Weights (First Layer, First Head)")
    plt.tight_layout()
    plt.savefig('bert_attention.png')
    print("\nSaved attention visualization as 'bert_attention.png'")

if __name__ == "__main__":
    print("Exploring BERT Tokenization and Embeddings")
    print("=" * 80)
    
    # Demo 1: Tokenization
    explore_tokenization()
    
    # Demo 2: Word embeddings and similarities
    explore_embeddings()
    
    # Demo 3: Attention visualization
    print("\nVisualizing BERT's attention...")
    visualize_attention()
    print("\nExploration complete!")
