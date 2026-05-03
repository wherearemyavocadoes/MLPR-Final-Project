import pandas as pd
import numpy as np
import re

# Configuration
INPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_filtered.csv"
OUTPUT_FEATURES_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/stage3_features_linguistic_activity.csv"

# Pre-compiled regex for linguistics
emotion_words = {'happy', 'sad', 'angry', 'upset', 'depressed', 'anxious', 'joy', 'fear', 'crying', 'suicide', 'kill', 'hate'}
emotion_pattern = re.compile(r'\b(?:' + '|'.join(emotion_words) + r')\b', re.IGNORECASE)

def extract_linguistic_features(df):
    """
    3. Linguistic Behavior Features (Lightweight Computation)
    - Word count
    - Sentence length
    - Emotional word density
    - Exclamation usage
    - Question usage
    """
    print("Extracting Linguistic Behavior Features...")
    
    # Fill nan to prevent errors
    texts = df['text_raw'].fillna("")
    
    # Word count (splitting by space)
    # +1 to handle single words without spaces
    df['feat_word_count'] = texts.str.count(' ') + 1
    # Fix instances where text is empty completely
    df.loc[texts == "", 'feat_word_count'] = 0
    
    # We use roughly 15 words per sentence as proxy if actual punctuation is stripped,
    # but since raw text keeps punctuation, we can split by .!?
    texts_with_punct = df['title'].fillna("") + " " + df['selftext'].fillna("")
    df['feat_sentence_count'] = texts_with_punct.str.count(r'[.!?]+')
    df.loc[df['feat_sentence_count'] == 0, 'feat_sentence_count'] = 1 # at least 1 sent
    
    df['feat_avg_sentence_length'] = df['feat_word_count'] / df['feat_sentence_count']
    
    # Emotion word count based on regex
    df['feat_emotion_word_count'] = texts_with_punct.apply(lambda x: len(emotion_pattern.findall(str(x))))
    df['feat_emotion_word_density'] = df['feat_emotion_word_count'] / df['feat_word_count'].replace(0, 1)
    
    # Exclamation & Question usage
    df['feat_exclamation_count'] = texts_with_punct.str.count(r'!')
    df['feat_question_count'] = texts_with_punct.str.count(r'\?')
    
    return df

def extract_activity_features(df):
    """
    4. Activity Behavior Features (Efficient Group-Based Computation)
    - Time between posts (grouped by user)
    - Posting frequency per day/week
    - Late-night posting ratio
    - Sudden burst posting behavior
    """
    print("Extracting Activity Behavior Features...")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Must sort by author & time before diff
    df = df.sort_values(['author', 'timestamp'])
    
    # Time between posts in HOURS
    df['feat_time_since_last_post_hrs'] = df.groupby('author')['timestamp'].diff().dt.total_seconds() / 3600.0
    df['feat_time_since_last_post_hrs'] = df['feat_time_since_last_post_hrs'].fillna(-1) # First post = -1
    
    # Late night posting (Defined as midnight to 5 AM)
    df['hour'] = df['timestamp'].dt.hour
    df['is_late_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    
    # Calculate historical late-night ratio per user up to the CURRENT post
    # Expanding sum over expanding count effectively gives expanding mean 
    # (requires grouping by author)
    print("Calculating rolling late-night ratio...")
    df['feat_late_night_ratio'] = df.groupby('author')['is_late_night'].expanding().mean().reset_index(0, drop=True)
    
    # Sudden Burst: If the time since last post is < 1 hour and they've posted before
    df['feat_is_burst'] = ((df['feat_time_since_last_post_hrs'] > 0) & (df['feat_time_since_last_post_hrs'] < 1.0)).astype(int)
    
    # Clean up temp cols
    df = df.drop(columns=['hour', 'is_late_night'])
    
    return df

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} posts after filtering.")
    
    # Extract Linguistic
    df = extract_linguistic_features(df)
    
    # Extract Activity
    df = extract_activity_features(df)
    
    # Select only the feature columns + keys to save
    feature_cols = ['author', 'timestamp'] + [col for col in df.columns if col.startswith('feat_')]
    
    print(f"Saving {len(feature_cols)} features to {OUTPUT_FEATURES_FILE}...")
    df[feature_cols].to_csv(OUTPUT_FEATURES_FILE, index=False)
    print("Done Linguistic & Activity!")

if __name__ == "__main__":
    main()
