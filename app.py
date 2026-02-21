import os
import torch
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =====================
# APP SETUP
# =====================
app = Flask(__name__)
CORS(app)

# =====================
# LOAD MODEL (ONCE)
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "science-bot")

print("⏳ Loading Science Bot Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"✅ Model Loaded on {device.upper()}")

# =====================
# AI FUNCTION
# =====================
def get_model_reply(question: str) -> str:
    prompt = (
        f"Explain the science concept '{question}' "
        f"in 5 to 7 bullet points using simple words. "
        f"Include one simple example."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=2,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.form.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please ask a science question."})

    reply = get_model_reply(user_msg)
    return jsonify({"reply": reply})

# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
