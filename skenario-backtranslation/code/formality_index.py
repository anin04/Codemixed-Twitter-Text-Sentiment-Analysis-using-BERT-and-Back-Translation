import pandas as pd
import re
from string import punctuation
import nltk
from nltk.corpus import words
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


nltk.download('words')

# Baca dataset CSV  # ganti dengan path/file Anda
input_files = ["Training_Data/training1.csv", "Testing_Data/test1.csv", "Validation_Data/validation1.csv"]
df_list = []
for file in input_files:
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)

    # Gabung semua dataframe menjadi satu
df = pd.concat(df_list, ignore_index=True)
# Siapkan stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Muat daftar kata dasar Sastrawi
with open('kata-dasar.txt', 'r', encoding='utf-8') as f:
    indo_dict = set(f.read().split())

# Muat kosakata bahasa Inggris NLTK
english_vocab = set(words.words())

def is_indonesian_word(word):
    # Bersihkan dan lower
    word = word.lower()
    word = re.sub(rf"[{punctuation}]", "", word)
    stemmed = stemmer.stem(word)
    return stemmed in indo_dict

def is_english_word(word):
    word = word.lower()
    word = re.sub(rf"[{punctuation}]", "", word)
    return word in english_vocab

ratios_combined = []

for tweet in df['emoji_in']:
    words_in_tweet = tweet.split()
    total_words = len(words_in_tweet)
    
    if total_words == 0:
        ratios_combined.append(0.0)
        continue
    
    recognized_count = 0
    for w in words_in_tweet:
        # Jika kata dikenali bahasa Indonesia atau Inggris
        if is_indonesian_word(w) or is_english_word(w):
            recognized_count += 1
    
    ratio = recognized_count / total_words
    ratios_combined.append(ratio)

df['ratio_baku_campuran'] = ratios_combined

# Rata-rata rasio seluruh file
avg_ratio_campuran = df['ratio_baku_campuran'].mean()

print("Rata-rata kebakuan campuran (ID/EN) untuk seluruh file:", avg_ratio_campuran)

# Simpan ke CSV (opsional)
df.to_csv('ori1.csv', index=False)
