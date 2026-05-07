import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "omarsabri8756/AraT5v2-XLSum-arabic-text-summarization"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

def summarize(text):
    text = "أعد صياغة النص التالي بشكل خبري مختصر: " + text

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=120,
            num_beams=6
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)
