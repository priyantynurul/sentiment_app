import streamlit as st
import pandas as pd
import numpy as np
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.externals import joblib

# Load custom NLP model and tokenizer
# @st.cache_resource
# # def load_model():
# #     model_name = "path/to/your/model"  # Change to your model path
# #     model = AutoModelForSequenceClassification.from_pretrained(model_name)
# #     tokenizer = AutoTokenizer.from_pretrained(model_name)
# #     return model, tokenizer

# # model, tokenizer = load_model()


import re
import string
from typing import List
from keras.models import load_model


custom_stopwords = {'yg', 'utk', 'cuman', 'deh', 'Btw', 'tapi', 'gua', 'gue', 'lo', 'lu',
    'kalo', 'trs', 'jd', 'nih', 'ntr', 'nya', 'lg', 'gk', 'ecusli', 'dpt',
    'dr', 'kpn', 'kok', 'kyk', 'donk', 'yah', 'u', 'ya', 'ga', 'km', 'eh',
    'sih', 'eh', 'bang', 'br', 'kyk', 'sll', 'rp', 'jt', 'kan', 'gpp', 'sm', 'usah',
    'mas', 'sob', 'thx', 'ato', 'jg', 'gw', 'wkwkwk', 'mak', 'haha', 'iy', 'k','t',
    'tp','haha', 'dg', 'dri', 'duh', 'ye', 'wkwk', 'syg', 'btw',
    'nerjemahin', 'gaes', 'guys', 'moga', 'kmrn', 'nemu', 'yukk',
    'wkwkw', 'klas', 'iw', 'ew', 'lho', 'sbnry', 'org', 'gtu', 'bwt',
    'krlga', 'clau', 'lbh', 'cpet', 'ku', 'wke', 'mba', 'mas', 'sdh', 'kmrn',
    'oi', 'spt', 'dlm', 'bs', 'krn', 'jgn', 'sapa', 'spt', 'sh', 'wakakaka',
    'sihhh', 'hehe', 'ih', 'dgn', 'la', 'kl', 'ttg', 'mana', 'kmna', 'kmn',
    'tdk', 'tuh', 'dah', 'kek', 'ko', 'pls', 'bbrp', 'pd', 'mah', 'dhhh',
    'kpd', 'tuh', 'kzl', 'byar', 'si', 'sii', 'cm', 'sy', 'hahahaha', 'weh','n',
    'dlu', 'tuhh','tpi','krn','kl','kbnykn','jd','lah',
    'dong','koq','pdhl','dg','tp','amp','wlpun','pst','tuk',
    'laaah','si','trs','tuh','cuyy','ehem','sih','di','d','kalee','jg','emg','wkwkwkw','anjritt','sih',
    'kok','mbok','nie','sj','ajg','ente','hm','yang','tau','kjmu','gimana','kayak','nggak','klo','dar',
    'aja','rb','emang','karna','gin','ngga','msh','iya','kah','pake','kayanya','gitu','doang','jadi','bangasal'}

def clean_indonesian_text(text: str) -> str:
    def remove_urls(text: str) -> str:
        """Menghapus URL dari teks"""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    def remove_usernames(text: str) -> str:
        """Menghapus username Twitter (@username)"""
        return re.sub(r'@\w+', '', text)

    def remove_hashtags(text: str) -> str:
        """Menghapus hashtag (#hashtag)"""
        return re.sub(r'#\w+', '', text)

    def remove_special_chars(text: str) -> str:
        """Menghapus karakter khusus dan emoji"""
        # Menghapus emoji dan karakter unicode khusus
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # simbol & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # bendera (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        # Menghapus karakter khusus seperti 'ðŸ'‡', 'â€¦'
        text = re.sub(r'[^\w\s\.,!?]', '', text)
        return text

    def clean_whitespace(text: str) -> str:
        """Membersihkan spasi berlebih"""
        # Mengganti multiple whitespace dengan single space
        text = re.sub(r'\s+', ' ', text)
        # Menghapus whitespace di awal dan akhir
        return text.strip()

    # Aplikasikan semua fungsi pembersihan
    text = text.lower()  # Ubah ke lowercase
    text = remove_urls(text)
    text = remove_usernames(text)
    text = remove_hashtags(text)
    text = remove_special_chars(text)
    text = clean_whitespace(text)

    return text

# Contoh penggunaan untuk dataset
def clean_dataset(texts: List[str]) -> List[str]:
    return [clean_indonesian_text(text) for text in texts]

def text_bersih(text):
    """
    Membersihkan teks dari URL, hashtag, dan karakter-karakter khusus

    Parameters:
    text (str): Teks yang akan dibersihkan

    Returns:
    str: Teks yang sudah dibersihkan
    """
    # Konversi ke lowercase
    text = text.lower()

    # Menghapus URL Twitter (termasuk format pic.twitter.com)
    text = re.sub(r'pic\.twitter\.com/\S+', '', text)

    # Menghapus URL umum
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Menghapus hashtag (kata yang diawali dengan #)
    text = re.sub(r'#\w+', '', text)

    # Menghapus mention (@username)
    text = re.sub(r'@\w+', '', text)

    # Menghapus emoji
    text = emoji.replace_emoji(text, '')

    # Menghapus karakter khusus dan tanda baca
    text = re.sub(r'[^\w\s]', '', text)

    # Menghapus angka
    text = re.sub(r'\d+', '', text)

    # Menghapus multiple spaces dan whitespace
    text = ' '.join(text.split())

    # Menghapus newline dan tab
    text = text.replace('\n', ' ').replace('\t', ' ')

    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    return text.strip()

def remove_stopwords(text, custom_stopwords=set()):
    stop_words = set(stopwords.words('indonesian'))
    # Gabungkan stopwords dari NLTK dengan stopword kustom
    stop_words = stop_words.union(custom_stopwords)
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def tokenize_tweet(text):
    if isinstance(text, str):
        return text.split()
    return []

def stemming_text(tokens):
    stemmer = StemmerFactory().create_stemmer()
    if isinstance(tokens, list):
        [stemmer.stem(token) for token in tokens]
        return " ".join(tokens)
    return tokens

# Preprocessing function
def preprocess_text(text):
    text_ = clean_indonesian_text(text)
    text__ = text_bersih(text_)
    text___ = remove_stopwords(text__)
    # inputs = tokenizer(text___, return_tensors="pt", truncation=True, padding=True)
    return text___


# Load the trained model (ensure model.pkl exists)
with open('tfidf_rf_tuned_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)


# Prediction function
def predict(text):
    inputs = preprocess_text(text)
    prediction = model.predict(inputs)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # predicted_label = torch.argmax(scores, dim=-1).item()
    # confidence = scores.max().item()

    return prediction

# Streamlit UI
st.title("Analisis Sentimen Pemilu 2019")
st.subheader("Mari ketahui bagaimana sentimen pemilih terhadap pemilu 2019")

# User input
user_input = st.text_area("Enter text:", "Type here...")

if st.button("Analyze"):
    label = predict(user_input)
     # Display the result
    st.markdown(f'<div class="subheader">Twit tersebut memiliki sentimen:</div>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color: #2a3d66;">${label}k</h3>', unsafe_allow_html=True)
