import os
import torch
import pandas as pd
import re
from transformers import MarianMTModel, MarianTokenizer

# Load CSV files
train_df = pd.read_csv('Training_Data/training1.csv')
test_df = pd.read_csv('Testing_Data/test1.csv')
validation_df = pd.read_csv('Validation_Data/validation1.csv')

# Add a 'dataset' column to track the source of the data (train, test, validation)
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
validation_df['dataset'] = 'validation'

# Combine the three datasets
combined_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)

# Check for the column name in the combined_df
print("Columns in combined_df:", combined_df.columns)

# Replace 'tweet' with the actual column name if it's different
text_column = 'tweet' if 'tweet' in combined_df.columns else 'your_actual_column_name'

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MarianMT models and tokenizers
model_id_en = 'Helsinki-NLP/opus-mt-id-en'
model_en_id = 'Helsinki-NLP/opus-mt-en-id'

tokenizer_id_en = MarianTokenizer.from_pretrained(model_id_en)
model_id_en = MarianMTModel.from_pretrained(model_id_en).to(device)

tokenizer_en_id = MarianTokenizer.from_pretrained(model_en_id)
model_en_id = MarianMTModel.from_pretrained(model_en_id).to(device)

# Define translation function
def translate(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors="pt", padding=True).to(device)
    translated_tokens = model.generate(**tokens)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_text[0]

# Function to back-translate with logging every 50 translations
def back_translate(text, idx):
    text_en = translate(text, tokenizer_id_en, model_id_en)
    text_id = translate(text_en, tokenizer_en_id, model_en_id)
    
    # Print every 50th translation and indicate the index of the data
    if idx % 50 == 0:
        print(f"Data ke-{idx}:")
        print(f"Original: {text}")
        print(f"Back-translated: {text_id}")
        print("-" * 50)
    
    return text_id

# Apply back-translation to the specified column in combined dataset
combined_df['backtranslated_tweet'] = [back_translate(text, idx) for idx, text in enumerate(combined_df[text_column])]

# Split the data back into train, test, and validation based on 'dataset' column
train_df = combined_df[combined_df['dataset'] == 'train'].drop(columns=['dataset'])
test_df = combined_df[combined_df['dataset'] == 'test'].drop(columns=['dataset'])
validation_df = combined_df[combined_df['dataset'] == 'validation'].drop(columns=['dataset'])

# Save the modified DataFrames back to CSV files
train_df.to_csv('training_BT_MT.csv', index=False)
test_df.to_csv('test_BT_MT.csv', index=False)
validation_df.to_csv('validation_BT_MT.csv', index=False)

print("Proses back-translation selesai dan file disimpan.")
