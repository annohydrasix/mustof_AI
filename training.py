import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re

# 🔧 Fungsi bantu untuk normalisasi teks
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # hilangkan tanda baca
    text = re.sub(r"\s+", " ", text).strip()     # hapus spasi ganda
    return text

# 📥 Load data latih
# Format CSV:
# kolom: pertanyaan,label
# label: 1 (valid), 0 (tidak valid)
df = pd.read_csv("./data/data_training.csv")
df['pertanyaan'] = df['pertanyaan'].fillna("").apply(normalize_text)
df['label'] = df['label'].fillna(0).astype(int)

# 🔢 TF-IDF vektorisasi
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['pertanyaan'])
y = df['label']

# 🧠 Train model klasifikasi
clf = LogisticRegression()
clf.fit(X, y)

# 💾 Simpan model dan vectorizer
joblib.dump(clf, "./data/validity_model.pkl")
joblib.dump(vectorizer, "./data/vectorizer.pkl")

print("✅ Model dan vectorizer berhasil disimpan!")
