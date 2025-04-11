from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("LumiOpen/Viking-33B")
model = AutoModelForCausalLM.from_pretrained("LumiOpen/Viking-33B")

def translate_finnish_to_english(finnish_text):
    # Create a translation prompt
    prompt = f"""Translate the following Finnish text to English:
Finnish: {finnish_text}
English:"""
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate translation
    outputs = model.generate(
        inputs["input_ids"],
        max_length=1024,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the English part (after "English:")
    try:
        return translation.split("English:")[1].strip()
    except IndexError:
        return translation.strip()

# Example usage
if __name__ == "__main__":
    finnish_text = "Hyvää päivää! Miten voit?"
    english_translation = translate_finnish_to_english(finnish_text)
    print(f"Finnish: {finnish_text}")
    print(f"English: {english_translation}") 