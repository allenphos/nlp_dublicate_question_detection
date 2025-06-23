import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Load once
def load_model(model_name="google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

# Check if string is a valid question
def is_question(text):
    return isinstance(text, str) and text.strip().endswith("?") and len(text.strip()) > 8

# Normalize text
def clean_text(text):
    return text.strip().lower().replace("  ", " ")

# Batched paraphrasing
def batch_paraphrase(text_list, tokenizer, model, device, batch_size=32, max_new_tokens=50):
    results = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(
            [f"paraphrase: {t}" for t in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Guarantee alignment: one paraphrase per input
        if len(decoded) != len(batch):
            print(f"⚠️ Warning: Output mismatch at batch {i}–{i+batch_size}")
            decoded = batch  # fallback: return original texts

        results.extend(decoded)
    return results
