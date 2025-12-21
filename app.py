import os
from flask import Flask, jsonify, request, session, render_template, redirect, url_for
from flask_cors import CORS
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo.errors import ServerSelectionTimeoutError

# 🔥 TRANSFORMERS IMPORTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

load_dotenv()

# ----------------- Flask App Setup -----------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback_secret")
app.config["MONGO_URI"] = os.getenv("MONGO_URI")

CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False
)

mongo = PyMongo(app)
app.db = mongo.cx[os.getenv("DB_NAME", "CHAT")]
app.users_collection = app.db["users"]
app.history_collection = app.db["history"]

# ----------------- LOAD FLAN-T5 MODEL -----------------
MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("✅ FLAN-T5 Model Loaded")

# ----------------- Helper -----------------
def safe_str(val):
    try:
        return str(val)
    except Exception:
        return repr(val)

# ----------------- AI REPLY FUNCTION -----------------
def get_model_reply(question: str) -> str:
    prompt = f"""
Explain the science concept '{question}' in 5 to 7 bullet points using simple words.
Include one simple example.
Answer accurately, do not hallucinate.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            min_new_tokens=120,
            max_new_tokens=250,
            num_beams=4,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            do_sample=False
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# ----------------- Routes -----------------
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("chat_page"))
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if "user_id" in session:
        return redirect(url_for("chat_page"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            return "Username and password required", 400

        if app.users_collection.find_one({"username": username}):
            return "Username already exists", 400

        hashed_pw = generate_password_hash(password)
        app.users_collection.insert_one({
            "username": username,
            "password": hashed_pw,
            "created_at": datetime.now(timezone.utc)
        })
        return redirect(url_for("home"))

    return render_template("signup.html")

@app.route("/login", methods=["POST"])
def login_page():
    username = request.form.get("username")
    password = request.form.get("password")

    user = app.users_collection.find_one({"username": username})
    if user and check_password_hash(user["password"], password):
        session["user_id"] = str(user["_id"])
        return redirect(url_for("chat_page"))

    return "Invalid credentials", 401

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("home"))

@app.route("/chat", methods=["GET", "POST"])
def chat_page():
    if "user_id" not in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        user_input = request.form.get("message", "").strip()
        if not user_input:
            return jsonify({"reply": "Empty message"}), 400

        reply = get_model_reply(user_input)

        try:
            app.history_collection.insert_one({
                "user_id": session["user_id"],
                "question": user_input,
                "answer": reply,
                "timestamp": datetime.now(timezone.utc)
            })
        except ServerSelectionTimeoutError:
            return jsonify({"reply": "MongoDB connection error"}), 500

        return jsonify({"reply": reply})

    return render_template("chat.html")

@app.route("/chat/history")
def chat_history():
    if "user_id" not in session:
        return jsonify([])

    history = list(
        app.history_collection
        .find({"user_id": session["user_id"]})
        .sort("timestamp", 1)
    )

    return jsonify([
        {"question": h["question"], "answer": h["answer"]}
        for h in history
    ])

# ----------------- Run -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
