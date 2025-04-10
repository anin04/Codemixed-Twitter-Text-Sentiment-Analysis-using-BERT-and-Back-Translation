import os
import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load CSV files
train_df = pd.read_csv("Training_Data/training1.csv")
test_df = pd.read_csv("Testing_Data/test1.csv")
validation_df = pd.read_csv("Validation_Data/validation1.csv")

# Tambahkan kolom 'dataset' untuk melacak asal data (train, test, validation)
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
validation_df['dataset'] = 'validation'

# Gabungkan ketiga dataset
combined_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)

# Define a function to remove text between colons (e.g., :pleading_face:)
#def remove_emojis(text):
    # This regex will find patterns like :word: and remove them
 #   return re.sub(r':[^:]*:', '', text)

# Apply the removal of emojis to the 'tweet' column for all data
# combined_df['cleaned_tweet'] = combined_df['tweet'].apply(remove_emojis)

# Cek apakah GPU tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load NLLB-200 models and tokenizers
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


# Define translation function for NLLB without src_lang and tgt_lang
def translate(text, tokenizer, model, tgt_lang_code):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)  # Get token ID for target language
    translated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_text[0]



# Function to back-translate with logging every 5 translations
def back_translate(text, idx):
    # Translate from Indonesian to English and back to Indonesian using NLLB-200
    text_id = translate(text, tokenizer, model, tgt_lang_code="eng_Latn")
    text_en = translate(text_id, tokenizer, model, tgt_lang_code="ind_Latn")
    
    # Print every 5th translation and indicate the index of the data
    if idx % 5 == 0:
        print(f"Data ke-{idx}:")
        print(f"Original: {text}")
        print(f"Back-translated: {text_en}")
        print("-" * 50)
    
    return text_en
 
# Apply back-translation to the combined dataset's 'cleaned_tweet' column
combined_df['backtranslated_tweet'] = [back_translate(text, idx) for idx, text in enumerate(combined_df['emoji_in'])]

# Pisahkan kembali data berdasarkan kolom 'dataset'
train_df = combined_df[combined_df['dataset'] == 'train'].drop(columns=  ['dataset'])
test_df = combined_df[combined_df['dataset'] == 'test'].drop(columns=['dataset'])
validation_df = combined_df[combined_df['dataset'] == 'validation'].drop(columns=['dataset'])

# Save the modified DataFrames back to CSV files
train_df.to_csv('Training_Data/BT_NLLB.csv', index=False)
test_df.to_csv('Testing_Data/BT_NLLB.csv', index=False)
validation_df.to_csv('Validation_Data/BT_NLLB.csv', index=False) 

print("Proses back-translation selesai dan file disimpan.")
