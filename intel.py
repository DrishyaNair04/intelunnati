# Import necessary libraries
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('openlm-research/open_llama_7b')
model = LlamaForCausalLM.from_pretrained('openlm-research/open_llama_7b')

# Define text generation function with adjustable parameters
def generate_text(input_text, max_length=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2, num_beams=5, no_repeat_ngram_size=2, early_stopping=True):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Test the function
if __name__ == "__main__":
    input_text = "Once upon a time"
    generated_text = generate_text(input_text)
    print(generated_text)