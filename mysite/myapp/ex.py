from transformers import AutoTokenizer, AutoModelForCausalLM
from googletrans import Translator  # Importing the library for translation

def main():
    print("Welcome to the BioGPT-based symptom analyzer!")
    print("Enter symptoms separated by commas, for example: 'headache, nausea, fever'.")

    symptoms = input("Enter symptoms: ").strip()

    if not symptoms:
        print("You did not enter any symptoms. Please try again.")
        return

    try:
        translator = Translator()

        # Translating symptoms into English
        symptoms_en = translator.translate(symptoms, src='ru', dest='en').text

        print("Loading the model and tokenizer... This might take a few seconds.")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
        model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

        print("Analyzing symptoms...")
        input_text = f"Patient symptoms: {symptoms_en}. Based on these symptoms, provide possible diagnoses:"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,  # Creativity balance
            do_sample=True,   # Enable sampling
            top_k=50,         # Limit to 50 likely tokens
            top_p=0.9,        # Nucleus sampling
            num_return_sequences=1
        )

        # Decoding the diagnosis in English
        diagnosis_en = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Translating the diagnosis back into Russian
        
        print(f"\nPossible diagnoses : {diagnosis_en}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()