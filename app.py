import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import os
import subprocess
import shutil

from transformers import pipeline
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="w11wo/indonesian-roberta-base-sentiment-classifier"
    )

sentiment_model = load_model()

# =========================
# NLTK SAFE
# =========================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# =========================
# KONFIGURASI GLOBAL
# =========================
DIREKTORI_SCRAPING = "data_scraping"
TOKEN_TWITTER = "0ed18874d5c0dcac089700328df1c556098c009b"  

if not os.path.exists(DIREKTORI_SCRAPING):
    os.makedirs(DIREKTORI_SCRAPING)


# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("📊 Sentiment Analysis Dashboard")

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "df" not in st.session_state:
    st.session_state.df = None

if "df_mentah" not in st.session_state:
    st.session_state.df_mentah = None

# =========================
# NAVIGATION FUNCTION
# =========================
def go_to(page):
    st.session_state.page = page

# =========================
# SIDEBAR
# =========================
menu_list = [
    "Home",
    "Scraping Data",
    "Input Data",
    "Filtering & Deduplikasi",
    "Preprocessing",
    "Labeling",
    "Modeling",
    "Evaluation",
    "Visualisasi"
]

menu = st.sidebar.radio(
    "📌 Navigasi",
    menu_list,
    index=menu_list.index(st.session_state.page)
)

st.session_state.page = menu

# =========================
# HOME
# =========================
if menu == "Home":

    st.markdown("---") 
    st.markdown("### Analisis opini publik dengan NLP & Machine Learning")  
    st.markdown(""" Dashboard ini membantu Anda menganalisis sentimen dari data teks secara end-to-end. 
                
    🔄 Alur Analisis 
    1. Input / Scraping Data 
    2. Filtering & Cleaning 
    3. Preprocessing 
    4. Labeling 
    5. Modeling 
    6. Visualisasi 
                
    Mulai dengan memilih sumber data: 
                """) 
    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🐦 Scraping Data")
        if st.button("🚀 Mulai Scraping"):
            go_to("Scraping Data")

    with col2:
        st.subheader("📥 Upload Data")
        if st.button("📂 Upload Dataset"):
            go_to("Input Data")

# =========================
# INPUT DATA
# =========================
elif menu == "Input Data":

    st.header("📥 Upload Dataset")

    file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

    valid = True  # kontrol

    if file is not None:

        nama_file = file.name.lower()

        try:
            if nama_file.endswith(".csv"):
                try:
                    df = pd.read_csv(file, encoding="utf-8")
                except:
                    try:
                        df = pd.read_csv(file, encoding="latin1")
                    except:
                        df = pd.read_csv(file, sep=";", encoding="latin1")

            elif nama_file.endswith(".xlsx"):
                df = pd.read_excel(file)

            else:
                st.error("❌ Format file tidak didukung")
                valid = False

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            valid = False

        # =========================
        # VALIDASI
        # =========================
        if valid:
            if df is None or df.empty:
                st.error("❌ Dataset kosong")
                valid = False

            elif len(df.columns) == 0:
                st.error("❌ Dataset tidak memiliki kolom")
                valid = False

            elif "full_text" not in df.columns:
                st.error(f"❌ Kolom 'full_text' tidak ditemukan.\nKolom: {list(df.columns)}")
                valid = False

        # =========================
        # SIMPAN
        # =========================
        if valid:
            st.session_state.df_mentah = df
            st.success(f"✅ Dataset berhasil ({len(df)} baris)")
            st.dataframe(df.head())

    else:
        st.info("📂 Upload file CSV atau Excel (.xlsx)")


    col1, col2 = st.columns(2)
    col1.button("⬅️ Back", on_click=go_to, args=("Home",))
    col2.button("➡️ Next", on_click=go_to, args=("Filtering & Deduplikasi",))

# ======================================================
# PENGUMPULAN DATA
# ======================================================
elif menu == "Scraping Data":
    st.subheader("🔎 Pengumpulan Data Twitter/X")

    st.markdown("""
    Pengambilan data tweet berdasarkan kata kunci utama.
    """)

    # =========================
    # INPUT KATA KUNCI
    # =========================
    kata_fraud = st.text_input(
        "Kata Kunci",
        "board of peace"
    )

    # =========================
    # PARAMETER
    # =========================
    batas_data = st.number_input(
        "Jumlah maksimum tweet per kata",
        min_value=100,
        max_value=10000,
        value=1000
    )

    tanggal_mulai = st.date_input("Tanggal Mulai")
    tanggal_akhir = st.date_input("Tanggal Akhir")

    # =========================
    # EKSEKUSI
    # =========================
    if st.button("🚀 Jalankan Scraping"):

        daftar_kata = [k.strip().lower() for k in kata_fraud.split(",")]

        os.makedirs(DIREKTORI_SCRAPING, exist_ok=True)

        mulai = tanggal_mulai.strftime("%Y-%m-%d")
        akhir = tanggal_akhir.strftime("%Y-%m-%d")

        semua_data = []

        st.info("Proses scraping berjalan...")

        for kata in daftar_kata:

            nama_file = f"data_{kata.replace(' ', '_')}.csv"

            query = (
                f'"{kata}" '
                f'since:{mulai} until:{akhir} '
                f'lang:id -is:retweet'
            )

            st.write(f"🔍 Query: **{query}**")

            command = (
                f'npx -y tweet-harvest@2.6.1 '
                f'-o "{nama_file}" '
                f'-s "{query}" '
                f'--tab "LATEST" '
                f'-l {batas_data} '
                f'--token "{TOKEN_TWITTER}"'
            )

            hasil = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )

            if hasil.returncode != 0:
                st.error(f"❌ Gagal scraping: {kata}")
                st.code(hasil.stderr)
                continue

            sumber = os.path.join(os.getcwd(), "tweets-data", nama_file)
            tujuan = os.path.join(DIREKTORI_SCRAPING, nama_file)

            if os.path.exists(sumber):
                shutil.move(sumber, tujuan)

                df = pd.read_csv(tujuan)
                semua_data.append(df)

                st.success(f"✅ {kata} berhasil ({len(df)} data)")
            else:
                st.warning(f"⚠️ Data tidak ditemukan untuk: {kata}")

        # =========================
        # GABUNG OTOMATIS
        # =========================
        if semua_data:
            df_final = pd.concat(semua_data, ignore_index=True)
            st.session_state.df_mentah = df_final

            st.success(f"🎉 Total data terkumpul: {len(df_final)}")
            st.dataframe(df_final.head())
        else:
            st.error("❌ Tidak ada data yang berhasil dikumpulkan")
        
    col1, col2 = st.columns(2)
    col1.button("⬅️ Back", on_click=go_to, args=("Home",))
    col2.button("➡️ Next", on_click=go_to, args=("Filtering & Deduplikasi",))

# ======================================================
# FILTER DAN DEDUPLIKASI DATA
# ======================================================
elif menu == "Filtering & Deduplikasi":

    st.subheader("📚 Filter dan Deduplikasi Data")

    st.markdown("""
    Tahap ini menghapus data duplikat dan melakukan filter dataset
    sebelum masuk ke preprocessing.
    """)

    df = st.session_state.df_mentah

    # =========================
    # VALIDASI DATA
    # =========================
    if df is None:
        st.warning("❗ Data belum tersedia. Silakan lakukan Upload atau Scraping terlebih dahulu.")
        st.stop()

    st.success(f"✅ Data tersedia: {len(df)} baris")
    st.dataframe(df.head())

    # =========================
    # PILIH KOLOM ID
    # =========================
    kolom_id = st.selectbox(
        "Pilih kolom ID unik (untuk deduplikasi)",
        options=df.columns,
        index=df.columns.get_loc("id_str") if "id_str" in df.columns else 0
    )

    # =========================
    # OPSI FILTER TAHUN (OPSIONAL)
    # =========================
    gunakan_filter_tahun = st.checkbox("Gunakan filter tahun")

    if gunakan_filter_tahun:
        tahun_penelitian = st.text_input("Tahun (contoh: 2025)", "2025")

    # =========================
    # EKSEKUSI
    # =========================
    if st.button("🚀 Jalankan Filtering"):

        df_bersih = df.copy()

        # ------------------------------
        # DEDUPLIKASI
        # ------------------------------
        before = len(df_bersih)

        df_bersih = df_bersih.drop_duplicates(
            subset=[kolom_id],
            keep="first"
        )

        after = len(df_bersih)

        st.write(f"🔹 Data sebelum deduplikasi: {before}")
        st.write(f"🔹 Data setelah deduplikasi: {after}")

        # ------------------------------
        # FILTER TAHUN (OPSIONAL)
        # ------------------------------
        if gunakan_filter_tahun:

            if "created_at" in df_bersih.columns:

                df_bersih = df_bersih[
                    df_bersih["created_at"]
                    .astype(str)
                    .str.contains(tahun_penelitian, na=False)
                ]

                st.write(f"🔹 Setelah filter tahun {tahun_penelitian}: {len(df_bersih)} data")

            else:
                st.warning("⚠️ Kolom 'created_at' tidak ditemukan, filter tahun dilewati")

        # =========================
        # VALIDASI KOLOM TEXT
        # =========================
        if "full_text" not in df_bersih.columns:
            st.error("❌ Kolom 'full_text' wajib ada untuk proses selanjutnya")
            st.stop()

        # =========================
        # SIMPAN KE DATA BERSIH
        # =========================
        st.session_state.df = df_bersih

        st.success("✅ Filtering & deduplikasi selesai")

        st.dataframe(df_bersih.head(10))

        # =========================
        # DOWNLOAD
        # =========================
        nama_output = "data_bersih.xlsx"
        df_bersih.to_excel(nama_output, index=False)

        with open(nama_output, "rb") as file:
            st.download_button(
                label="⬇️ Download Data Bersih",
                data=file,
                file_name=nama_output,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # =========================
    # NAVIGASI
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.button("⬅️ Back to Input Data", on_click=go_to, args=("Input Data",))
    col2.button("⬅️ Back to Scrapping Data", on_click=go_to, args=("Scrapping Data",))
    col3.button("➡️ Next", on_click=go_to, args=("Preprocessing",))
    
# =========================
# PREPROCESSING
# =========================
elif menu == "Preprocessing":

    st.header("🧹 Preprocessing Text")

    df = st.session_state.df

    # =========================
    # VALIDASI
    # =========================
    if df is None:
        st.warning("❗ Lakukan filtering terlebih dahulu")
        st.stop()

    if "full_text" not in df.columns:
        st.error("❌ Kolom 'full_text' tidak ditemukan")
        st.stop()

    st.write(f"Jumlah data: {len(df)}")

    # =========================
    # SETUP NLP
    # =========================
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())

    stemmer = StemmerFactory().create_stemmer()

    # tambahan stopwords
    stopwords.update([
        "yang","dan","di","ke","dari","ini","itu","untuk","dengan","atau",
        "karena","ada","jadi","sudah","belum","akan","bisa","harus","lebih",
        "lagi","juga","agar","soal","tentang","mengenai","terkait","hal",
        "sebuah","para","semua","setiap","menurut","katanya","kayaknya",
        "sepertinya","mungkin","entah","aja","dong","nih","deh","sih","kok",
        "lah","pun","yah","kan","banget","kayak","gitu","begitu","malah",
        "justru","udah","sampe","ga","gak","nggak","engga","iya","wkwk",
        "haha","hehe","lol","anjir","anjay","astaga","buset","guys",
        "bang","kak","bro","sis","min","admin","gan","viral","rame",
        "update","rt","tweet","post","komen","reply","link","video",
        "orang","banyak","apa","siapa","kenapa","bagaimana","hari",
        "tahun","bulan","kemarin","sekarang","ok","oke","nah","loh","yg"
    ])

    # =========================
    # FUNCTION PIPELINE
    # =========================
    def cleaningText(text):
        text = str(text)
        text = re.sub(r"http\S+"," ",text)
        text = re.sub(r"@\w+"," ",text)
        text = re.sub(r"#"," ",text)
        text = re.sub(r"\d+"," ",text)
        text = text.translate(str.maketrans("","",string.punctuation))
        text = re.sub(r"[^a-zA-Z\s]"," ",text)
        return re.sub(r"\s+"," ",text).strip()

    def casefoldingText(text):
        return text.lower()

    def tokenizingText(text):
        return nltk.word_tokenize(text)

    def filteringText(tokens):
        return [word for word in tokens if len(word) > 2]

    def stemmingText(tokens):
        sentence = " ".join(tokens)
        return stemmer.stem(sentence).split()

    def stopwordRemoval(tokens):
        return [word for word in tokens if word not in stopwords]

    def toSentence(words):
        return " ".join(words)

    # =========================
    # EKSEKUSI
    # =========================
    if st.button("🚀 Jalankan Preprocessing"):

        progress = st.progress(0)

        df["cleaning"] = df["full_text"].apply(cleaningText)
        progress.progress(15)

        df["casefolding"] = df["cleaning"].apply(casefoldingText)
        progress.progress(30)

        df["tokenizing"] = df["casefolding"].apply(tokenizingText)
        progress.progress(45)

        df["filtering"] = df["tokenizing"].apply(filteringText)
        progress.progress(60)

        df["stemming"] = df["filtering"].apply(stemmingText)
        progress.progress(75)

        df["stopword_removal"] = df["stemming"].apply(stopwordRemoval)
        progress.progress(90)

        df["clean_text"] = df["stopword_removal"].apply(toSentence)
        progress.progress(100)

        # simpan ke session
        st.session_state.df = df

        st.success("✅ Preprocessing selesai")
        st.dataframe(df[["full_text", "clean_text"]].head())

        # =========================
        # DOWNLOAD
        # =========================
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Hasil Preprocessing",
            data=csv,
            file_name="dataset_preprocessing.csv",
            mime="text/csv"
        )

    col1, col2 = st.columns(2)
    col1.button("⬅️ Back", on_click=go_to, args=("Filtering & Deduplikasi",))
    col2.button("➡️ Next", on_click=go_to, args=("Labeling",))

# =========================
# LABELING
# =========================
elif menu == "Labeling":

    st.header("🏷️ Auto Label Sentiment")

    df = st.session_state.df

    # =========================
    # VALIDASI
    # =========================
    if df is None:
        st.warning("❗ Lakukan preprocessing terlebih dahulu")
        st.stop()

    if "clean_text" not in df.columns:
        st.error("❌ Kolom 'clean_text' tidak ditemukan")
        st.stop()

    st.write(f"Jumlah data: {len(df)}")

    # =========================
    # LOAD MODEL (CACHE)
    # =========================
    @st.cache_resource
    def load_model():
        return pipeline(
            "sentiment-analysis",
            model="w11wo/indonesian-roberta-base-sentiment-classifier"
        )

    sentiment_model = load_model()

    # =========================
    # FUNCTION LABELING
    # =========================
    def auto_label(text):
        try:
            result = sentiment_model(text[:512])[0]
            return result["label"].lower()
        except:
            return "neutral"

    # =========================
    # EKSEKUSI
    # =========================
    if st.button("🚀 Jalankan Auto Labeling"):

        progress = st.progress(0)
        results = []

        total = len(df)

        for i, text in enumerate(df["clean_text"]):
            results.append(auto_label(text))

            # update progress
            progress.progress((i + 1) / total)

        df["label"] = results

        st.success("✅ Labeling selesai")

        # =========================
        # DISTRIBUSI AWAL
        # =========================
        st.subheader("Distribusi Label (Awal)")
        st.write(df["label"].value_counts())

        # =========================
        # SIMPAN ALL DATA
        # =========================
        csv_all = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Dataset (All Label)",
            data=csv_all,
            file_name="dataset_labeled_all.csv",
            mime="text/csv"
        )

        # =========================
        # HAPUS NETRAL
        # =========================
        df_non_netral = df[df["label"] != "neutral"].reset_index(drop=True)

        st.subheader("Distribusi Label (Tanpa Neutral)")
        st.write(df_non_netral["label"].value_counts())

        # =========================
        # TOTAL POSITIVE + NEGATIVE
        # =========================
        total_pos_neg = df_non_netral["label"].value_counts().sum()

        st.write(f"Total Positif & Negatif: {total_pos_neg}")

        # =========================
        # SIMPAN NON NETRAL
        # =========================
        csv_non = df_non_netral.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Dataset (Non Neutral)",
            data=csv_non,
            file_name="dataset_labeled_non_netral.csv",
            mime="text/csv"
        )

        # =========================
        # SIMPAN KE SESSION
        # =========================
        st.session_state.df = df_non_netral

        st.dataframe(df_non_netral.head())

    # =========================
    # NAVIGASI
    # =========================
    col1, col2 = st.columns(2)
    col1.button("⬅️ Back", on_click=go_to, args=("Preprocessing",))
    col2.button("➡️ Next", on_click=go_to, args=("Modeling",))

# =========================
# MODELING
# =========================
elif menu == "Modeling":

    st.header("🤖 Modeling & Model Selection")

    df = st.session_state.df

    if df is None:
        st.warning("❗ Lakukan labeling terlebih dahulu")
        st.stop()

    # =========================
    # ENCODING
    # =========================
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    # =========================
    # SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # =========================
    # TF-IDF
    # =========================
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2))

    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    # =========================
    # SMOTE
    # =========================
    st.subheader("Distribusi Sebelum SMOTE")
    st.write(pd.Series(le.inverse_transform(y_train)).value_counts())

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    st.subheader("Distribusi Setelah SMOTE")
    st.write(pd.Series(le.inverse_transform(y_train)).value_counts())

    # =========================
    # MODEL LIST
    # =========================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "SVM": LinearSVC(class_weight="balanced"),
        "Naive Bayes": MultinomialNB()
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    st.subheader("📊 Cross Validation Result")

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=skf,
            scoring="f1_macro"
        )

        results[name] = scores.mean()

    results_df = pd.DataFrame(list(results.items()), columns=["Model", "F1 Score"])
    st.dataframe(results_df)

    # =========================
    # PILIH MODEL TERBAIK
    # =========================
    best_model_name = max(results, key=results.get)
    st.success(f"🏆 Model Terbaik: {best_model_name}")

    final_model = models[best_model_name]
    final_model.fit(X_train, y_train)

    # =========================
    # SIMPAN KE SESSION
    # =========================
    st.session_state.model = final_model
    st.session_state.tfidf = tfidf
    st.session_state.label_encoder = le
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

    st.success("✅ Model berhasil disimpan")

    col1, col2 = st.columns(2)
    col1.button("⬅️ Back", on_click=go_to, args=("Labeling",))
    col2.button("➡️ Next", on_click=go_to, args=("Evaluation",))

# =========================
# EVALUATION
# =========================
elif menu == "Evaluation":

    st.header("📊 Model Evaluation")

    if "model" not in st.session_state:
        st.warning("❗ Jalankan modeling terlebih dahulu")
        st.stop()

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    le = st.session_state.label_encoder

    # =========================
    # PREDIKSI
    # =========================
    y_pred = model.predict(X_test)

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        output_dict=True
    )

    st.subheader("📄 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax
    )

    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.button("⬅️ Back", on_click=go_to, args=("Modeling",))
    col2.button("➡️ Next", on_click=go_to, args=("Visualisasi",))

# =========================
# VISUALISASI
# =========================
elif menu == "Visualisasi":

    st.header("📊 Visualisasi Sentimen")

    df = st.session_state.df

    # =========================
    # VALIDASI
    # =========================
    if df is None:
        st.warning("❗ Data belum tersedia")
        st.stop()

    if "label" not in df.columns:
        st.error("❌ Kolom 'label' tidak ditemukan")
        st.stop()

    # =========================
    # DISTRIBUSI SENTIMEN
    # =========================
    st.subheader("📌 Distribusi Sentimen")

    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="label", ax=ax1)
    ax1.set_title("Distribusi Sentimen")
    st.pyplot(fig1)

    st.write(df["label"].value_counts())

    # =========================
    # PIE CHART (PENTING UNTUK SKRIPSI)
    # =========================
    st.subheader("📊 Proporsi Sentimen")

    fig2, ax2 = plt.subplots()
    df["label"].value_counts().plot.pie(
        autopct="%1.1f%%",
        ax=ax2
    )
    ax2.set_ylabel("")
    ax2.set_title("Proporsi Sentimen")
    st.pyplot(fig2)

    # =========================
    # TIMELINE SENTIMEN
    # =========================
    if "created_at" in df.columns:

        st.subheader("📈 Tren Sentimen (Time Series)")

        try:
            df["created_at"] = pd.to_datetime(
                df["created_at"],
                errors="coerce"
            )

            timeline = df.groupby(
                [df["created_at"].dt.date, "label"]
            ).size().unstack().fillna(0)

            fig3, ax3 = plt.subplots(figsize=(12,5))
            timeline.plot(ax=ax3)
            ax3.set_title("Trend Sentimen Publik")
            st.pyplot(fig3)

        except:
            st.warning("⚠️ Format tanggal tidak valid")

    # =========================
    # WORDCLOUD PER SENTIMEN
    # =========================
    st.subheader("☁️ WordCloud per Sentimen")

    for label in df["label"].unique():

        text = " ".join(df[df["label"] == label]["clean_text"])

        if text.strip() == "":
            continue

        wc = WordCloud(width=800, height=400).generate(text)

        fig4, ax4 = plt.subplots()
        ax4.imshow(wc)
        ax4.axis("off")
        ax4.set_title(f"WordCloud: {label}")

        st.pyplot(fig4)

    # =========================
    # TOP WORD FREQUENCY (PENTING)
    # =========================
    st.subheader("🔝 Kata Paling Sering Muncul")

    all_text = " ".join(df["clean_text"])
    words = all_text.split()

    freq = pd.Series(words).value_counts().head(15)

    fig5, ax5 = plt.subplots()
    freq.plot(kind="bar", ax=ax5)
    ax5.set_title("Top 15 Kata")
    st.pyplot(fig5)

    # =========================
    # SENTIMEN VS PANJANG TEKS
    # =========================
    st.subheader("📏 Panjang Teks vs Sentimen")

    df["text_length"] = df["clean_text"].apply(len)

    fig6, ax6 = plt.subplots()
    sns.boxplot(data=df, x="label", y="text_length", ax=ax6)
    ax6.set_title("Distribusi Panjang Teks")
    st.pyplot(fig6)

    # =========================
    # DOWNLOAD DATA
    # =========================
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Dataset Final",
        data=csv,
        file_name="dataset_final.csv",
        mime="text/csv"
    )

    # =========================
    # NAVIGASI
    # =========================
    col1, col2 = st.columns(2)
    col1.button("⬅️ Back", on_click=go_to, args=("Evaluation",))