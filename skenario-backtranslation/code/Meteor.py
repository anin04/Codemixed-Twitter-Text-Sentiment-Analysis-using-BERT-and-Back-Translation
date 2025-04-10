import pandas as pd
from tqdm import tqdm
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.translate.meteor_score import meteor_score
import emoji

nltk.download('wordnet')

# Load the original and backtranslated data
original_train = pd.read_csv('Raw_Data/trainingOri.csv')
original_val = pd.read_csv('Raw_Data/validationOri.csv')
original_test = pd.read_csv('Raw_Data/testOri.csv')
backtranslated_train = pd.read_csv('Training_Data/BT_GoogleAPI.csv')
backtranslated_val = pd.read_csv('Validation_Data/BT_GoogleAPI.csv')
backtranslated_test = pd.read_csv('Testing_Data/BT_GoogleAPI.csv')

# Ensure that all dataframes have the same length and columns
assert len(original_train) == len(backtranslated_train), "Mismatch in the number of rows in training data"
assert len(original_val) == len(backtranslated_val), "Mismatch in the number of rows in validation data"
assert len(original_test) == len(backtranslated_test), "Mismatch in the number of rows in testing data"
assert 'tweet' in original_train.columns and 'backtranslated_tweet' in backtranslated_train.columns, "Column 'tweet' not found in training dataframes"
assert 'tweet' in original_val.columns and 'backtranslated_tweet' in backtranslated_val.columns, "Column 'tweet' not found in validation dataframes"
assert 'tweet' in original_test.columns and 'backtranslated_tweet' in backtranslated_test.columns, "Column 'tweet' not found in testing dataframes"

# Initialize Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function to preprocess text
def preprocess_text(text):
    # Convert emoji to text
    text = emoji.demojize(text)
    # Tokenize and apply stemming
    return [stemmer.stem(word) for word in text.split()]

# Function to calculate METEOR score for a given pair of dataframes
def calculate_meteor_scores(original_df, backtranslated_df, description="Processing"):
    meteor_scores = []
    original_tweets = []
    backtranslated_tweets = []

    # Use tqdm for progress bar
    for i in tqdm(range(len(original_df)), desc=description):
        original_text = original_df.iloc[i]['tweet']
        backtranslated_text = backtranslated_df.iloc[i]['backtranslated_tweet']

        original_tweets.append(original_text)
        backtranslated_tweets.append(backtranslated_text)

        tokenized_reference = preprocess_text(original_text)
        tokenized_hypothesis = preprocess_text(backtranslated_text)

        score = meteor_score([tokenized_reference], tokenized_hypothesis)
        meteor_scores.append(score)
    
    return pd.DataFrame({
        'Original_Tweet': original_tweets,
        'Backtranslated_Tweet': backtranslated_tweets,
        'METEOR_score': meteor_scores
    })

# Calculate METEOR scores for training, validation, and testing sets with progress bars
train_meteor_df = calculate_meteor_scores(original_train, backtranslated_train, description="Calculating Training METEOR Scores")
val_meteor_df = calculate_meteor_scores(original_val, backtranslated_val, description="Calculating Validation METEOR Scores")
test_meteor_df = calculate_meteor_scores(original_test, backtranslated_test, description="Calculating Testing METEOR Scores")

# Calculate average METEOR scores for each set
average_meteor_train = train_meteor_df['METEOR_score'].mean()
average_meteor_val = val_meteor_df['METEOR_score'].mean()
average_meteor_test = test_meteor_df['METEOR_score'].mean()

# Combine all results into one dataframe
combined_meteor_df = pd.concat([train_meteor_df, val_meteor_df, test_meteor_df], keys=['Training', 'Validation', 'Testing']).reset_index(level=0).rename(columns={'level_0': 'Set'})

# Calculate combined average METEOR score
combined_average_meteor = combined_meteor_df['METEOR_score'].mean()

# Add average scores to the dataframe
combined_meteor_df['Average_METEOR_Train'] = average_meteor_train
combined_meteor_df['Average_METEOR_Validation'] = average_meteor_val
combined_meteor_df['Average_METEOR_Testing'] = average_meteor_test
combined_meteor_df['Combined_Average_METEOR'] = combined_average_meteor

# Save the result to a new CSV file
combined_meteor_df.to_csv('Meteor/google.csv', index=False)

# Print the average METEOR scores
print(f"Average METEOR score for Training Set: {average_meteor_train}")
print(f"Average METEOR score for Validation Set: {average_meteor_val}")
print(f"Average METEOR score for Testing Set: {average_meteor_test}")
print(f"Combined Average METEOR score: {combined_average_meteor}")

print("METEOR scores calculated and saved")
