from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

print("⬇️ Downloading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    use_safetensors=False   # 🔥 IMPORTANT
)

SAVE_PATH = "science-bot"

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print("✅ Model saved correctly (BIN format)")
