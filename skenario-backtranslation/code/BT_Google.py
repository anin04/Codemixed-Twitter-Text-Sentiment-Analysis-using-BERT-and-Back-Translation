import os
from google.cloud import translate_v2 as translate
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\S2-SMT 2\tesis konsul\code baruu\inbound-guru-438013-m3-e84f9af642f5.json"
translate_client = translate.Client()

# Function to perform back translation
def back_translate(text, src_lang="id", intermediate_lang="en"):
    try:
        # Translate from source to intermediate language (Indonesian to English)
        result = translate_client.translate(text, target_language=intermediate_lang, source_language=src_lang)
        translated_text = result['translatedText']
        
        # Translate back from intermediate language to source language (English to Indonesian)
        result_back = translate_client.translate(translated_text, target_language=src_lang, source_language=intermediate_lang)
        back_translated_text = result_back['translatedText']
        
        return back_translated_text
    except Exception as e:
        print(f"Error translating text: {text} - {e}")
        return text  # Return original text if there's an error

# Function to perform concurrent back translation while preserving order
def perform_back_translation(df, column_name, max_workers=100):
    # Dictionary to store results with indexes to maintain order
    translations = {}
    
    # Set up a ThreadPoolExecutor to make concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each row in the column to be translated
        futures = {executor.submit(back_translate, text): idx for idx, text in enumerate(df[column_name])}
        
        # Collect results as they complete
        for future in as_completed(futures):
            idx = futures[future]
            translations[idx] = future.result()
    
    # Map back to DataFrame to ensure order is preserved
    ordered_translations = [translations[i] for i in range(len(df))]
    return ordered_translations

# Load datasets
train_df = pd.read_csv("Training_Data/training1.csv")
test_df = pd.read_csv("Testing_Data/test1.csv")
validation_df = pd.read_csv("Validation_Data/validation1.csv")

# Perform concurrent back translation on a specific column (e.g., 'tweet')
train_df['backtranslated_tweet'] = perform_back_translation(train_df, 'emoji_in')
test_df['backtranslated_tweet'] = perform_back_translation(test_df, 'emoji_in')
validation_df['backtranslated_tweet'] = perform_back_translation(validation_df, 'emoji_in')

# Save the modified DataFrames to new CSV files
train_df.to_csv('Training_Data/BT_GoogleAPI.csv', index=False)
test_df.to_csv('Testing_Data/BT_GoogleAPI.csv', index=False)
validation_df.to_csv('Validation_Data/BT_GoogleAPI.csv', index=False)
