import streamlit as st
import random
from nlp_utils import load_dataset, get_best_response

st.set_page_config(page_title="Mustof_AI - Chatbot Islami", page_icon="🕌")

# Navigasi via sidebar
page = st.sidebar.selectbox("Navigasi", ["🕌 Chatbot", "ℹ️ Tentang Aplikasi"])

if page == "🕌 Chatbot":
    st.title("🕌 Mustof_AI - Chatbot Islami")
    st.write("Assalamualaikum, Silakan ketik pertanyaan Anda seputar ajaran Islam di bawah ini.")

    # Load model dan dataset
    questions, answers, question_embeddings, contexts = load_dataset()

    # Tampilkan rekomendasi pertanyaan
    st.markdown("#### 📌 Contoh Pertanyaan:")
    sample_questions = random.sample(questions, k=5)
    for q in sample_questions:
        st.markdown(f"- {q}")

    # Inisialisasi riwayat chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Form untuk input pertanyaan
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Pertanyaan Anda:", "")
        submit = st.form_submit_button("Kirim")

    # Logika percakapan
    if submit and user_input:
        if user_input.lower() == "exit":
            st.session_state.chat_history.append(("Anda", user_input))
            st.session_state.chat_history.append(("Mustof_AI", "Wa'alaikumussalam. Semoga harimu berkah!"))
        else:
            response = get_best_response(user_input, questions, answers, question_embeddings, contexts)
            st.session_state.chat_history.append(("Anda", user_input))
            st.session_state.chat_history.append(("Mustof_AI", response))

    # Riwayat Obrolan
    st.markdown("---")
    st.subheader("🗨️ Riwayat Obrolan")

    for speaker, message in st.session_state.chat_history:
        if speaker == "Anda":
            st.markdown(f"**🧑 {speaker}:** {message}")
        else:
            st.markdown(f"**🤖 {speaker}:** {message}")

    # Tombol untuk hapus riwayat
    if st.button("🗑️ Clear Riwayat"):
        st.session_state.chat_history = []
        st.rerun()

elif page == "ℹ️ Tentang Aplikasi":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
Mustof_AI adalah chatbot Islami yang dibangun menggunakan pendekatan Semantic Search dengan SBERT (Sentence-BERT) untuk memahami dan menjawab pertanyaan dalam Bahasa Indonesia secara kontekstual.

🔧 **Teknologi yang Digunakan**:

- 🧠 **Natural Language Processing (NLP)**  
  Untuk memahami bahasa alami dan menangani pertanyaan berbasis teks.

- 🔍 **Sentence-BERT (SBERT)**  
  Model berbasis Transformer yang di-fine-tune khusus untuk Bahasa Indonesia agar dapat menghitung kemiripan semantik antar kalimat.

- ⚙️ **Semantic Similarity & Cosine Similarity**  
  Untuk mencari jawaban yang paling relevan dari dataset berdasarkan makna, bukan hanya kata kunci.

- 🐍 **Python**  
  Bahasa pemrograman utama dalam pengembangan seluruh sistem.

- 📦 **HuggingFace Transformers & SentenceTransformers**  
  Library utama untuk mengelola model SBERT dan melakukan embedding kalimat.

- 🔗 **PyTorch**  
  Backend dari model SBERT untuk pemrosesan tensor dan inference.

- 📊 **Pandas**  
  Digunakan untuk memuat dan memproses dataset dalam format CSV.

- 🌐 **Streamlit**  
  Untuk membangun antarmuka aplikasi chatbot berbasis web secara interaktif dan ringan.

📚 **Topik yang Dapat Ditanyakan**:

- Rukun Islam & Rukun Iman  
- Shalat Wajib & Sunnah  
- Zakat, Puasa, dan Ibadah lainnya  
- Al-Qur’an dan Hadis  
- Akhlak, Adab, dan Sejarah Nabi

👨‍💻 **Dibuat Oleh**:

- 🧑 Nama: Fikri Khoiruddin  
- 🏫 Universitas: Universitas Indraprasta PGRI  
- 📅 Tahun: 2025  
- 📬 Email: fikrikhoiruddin28@gmail.com
    """)
