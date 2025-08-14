from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
import joblib

# Load model klasifikasi pertanyaan valid
validity_clf = joblib.load("./validity_model.pkl")
vectorizer = joblib.load("./vectorizer.pkl")

# Load pre-trained Sentence-BERT multilingual model
model = SentenceTransformer('indobenchmark/indobert-base-p1')

# Normalisasi teks (untuk preprocessing)
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Hilangkan tanda baca
    text = re.sub(r"\s+", " ", text).strip()     # Hapus spasi ganda
    return text

# Fungsi untuk memeriksa apakah pertanyaan valid
def is_valid_question(text):
    X = vectorizer.transform([text])
    prediction = validity_clf.predict(X)
    return prediction[0] == 1

# Fungsi untuk memuat dataset dan membuat embedding pertanyaan
def load_dataset(path='./faq.csv'):
    df = pd.read_csv(path)
    questions = df['pertanyaan'].fillna("").tolist()
    answers = df['jawaban'].fillna("").tolist()
    contexts = df['konteks'].fillna("").tolist()
    question_embeddings = model.encode([normalize_text(q) for q in questions], convert_to_tensor=True)
    return questions, answers, question_embeddings, contexts

# Fungsi utama untuk mencari jawaban terbaik berdasarkan input
def get_best_response(user_input, questions, answers, question_embeddings, contexts, threshold=0.8):
    if user_input.strip().lower() in ["apa", "siapa", "kenapa", "bagaimana", "tolong", "halo", "hai"]:
        return "Pertanyaan Anda terlalu umum. Mohon tulis lebih lengkap agar saya bisa membantu."

    user_input_clean = normalize_text(user_input)
    user_embedding = model.encode(user_input_clean, convert_to_tensor=True)

    # Filter berdasarkan konteks jika cocok ditemukan dalam input
    matched_context = None
    for ctx in contexts:
        if ctx.lower() in user_input_clean:
            matched_context = ctx
            break

    if matched_context:
        filtered_data = [
            (q, a) for q, a, c in zip(questions, answers, contexts)
            if c.lower() == matched_context.lower()
        ]
    else:
        filtered_data = [(q, a) for q, a in zip(questions, answers)]

    if not filtered_data:
        return "Maaf, tidak ada data yang relevan ditemukan."

    filtered_questions = [q for q, _ in filtered_data]
    filtered_answers = [a for _, a in filtered_data]

    if re.fullmatch(r"[a-zA-Z]+\?*", user_input.strip()) and len(user_input.strip()) <= 6:
        return "Pertanyaan Anda terlalu singkat. Mohon perjelas kembali."

    filtered_embeddings = model.encode(
        [normalize_text(q) for q in filtered_questions],
        convert_to_tensor=True
    )
    cosine_scores = util.pytorch_cos_sim(user_embedding, filtered_embeddings)
    best_match_index = cosine_scores.argmax().item()
    best_score = cosine_scores[0][best_match_index].item()
 
    if best_score < threshold:
        return "Maaf, saya belum mengerti pertanyaan Anda."
    return filtered_answers[best_match_index]


