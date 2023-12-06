from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


def translate_english_to_french(english_sentence):
    input_text = "translate English to French: " + english_sentence

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_new_tokens = 40)

    french_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return french_translation
