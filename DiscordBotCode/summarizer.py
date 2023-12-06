from transformers import BartTokenizer, BartForConditionalGeneration


def summarize_text(text) -> str:
    # Load pre-trained model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Encode the text input and generate summary
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
