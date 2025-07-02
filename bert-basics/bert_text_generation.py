from transformers import BertTokenizer, BertLMHeadModel
import torch

def load_generation_model():
    """Load BERT model and tokenizer for text generation."""
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertLMHeadModel.from_pretrained(model_name)
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=50, temperature=1.0, top_k=50, top_p=0.95):
    """Generate text continuation using BERT."""
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length + len(input_ids[0]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    # Load model and tokenizer
    print("Loading BERT model for text generation...")
    model, tokenizer = load_generation_model()
    
    # Example prompts
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a land far away",
        "The key to success is",
        "If I could travel anywhere, I would go to"
    ]
    
    # Generate text for each prompt
    print("\nText Generation with BERT")
    print("=" * 80)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 80)
        
        # Generate multiple samples with different temperatures
        for temp in [0.7, 1.0, 1.3]:
            generated = generate_text(
                prompt,
                model,
                tokenizer,
                max_length=50,
                temperature=temp
            )
            print(f"Temperature {temp}: {generated}")
            print("-" * 80)

if __name__ == "__main__":
    main()
