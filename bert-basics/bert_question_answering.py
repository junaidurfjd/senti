from transformers import BertTokenizer, BertForQuestionAnswering
import torch

def load_qa_model():
    """Load pre-trained BERT model and tokenizer for question answering."""
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    return model, tokenizer

def answer_question(question, context, model, tokenizer):
    """Answer a question based on the given context using BERT."""
    # Encode the question and context
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    
    # Get model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the start and end scores for the answer
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    # Convert token IDs to actual tokens
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end]
        )
    )
    
    return answer

def main():
    # Load model and tokenizer
    print("Loading BERT QA model and tokenizer...")
    model, tokenizer = load_qa_model()
    
    # Example context and questions
    context = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    Constructed from 1887 to 1889, it was initially criticized by some of France's leading 
    artists and intellectuals for its design, but it has become a global cultural icon of 
    France and one of the most recognizable structures in the world.
    """
    
    questions = [
        "Who designed the Eiffel Tower?",
        "When was the Eiffel Tower constructed?",
        "Where is the Eiffel Tower located?"
    ]
    
    # Answer questions
    print("\nQuestion Answering with BERT:")
    print("Context:", context.strip())
    print("-" * 80)
    
    for question in questions:
        answer = answer_question(question, context, model, tokenizer)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("-" * 80)

if __name__ == "__main__":
    main()
