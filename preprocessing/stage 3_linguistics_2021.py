import pandas as pd
import numpy as np
import re

INPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_filtered.csv"
OUTPUT_FEATURES_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/stage3_features_linguistic_activity_2021.csv"

emotion_words = {'happy', 'sad', 'angry', 'upset', 'depressed', 'anxious', 'joy', 'fear', 'crying', 'suicide', 'kill', 'hate'}
emotion_pattern = re.compile(r'\b(?:' + '|'.join(emotion_words) + r')\b', re.IGNORECASE)

def extract_linguistic_features(df):
    print("Extracting Linguistic Behavior Features...")
    texts = df['text_raw'].fillna("")
    df['feat_word_count'] = texts.str.count(' ') + 1
    df.loc[texts == "", 'feat_word_count'] = 0
    texts_with_punct = df['title'].fillna("") + " " + df['selftext'].fillna("")
    df['feat_sentence_count'] = texts_with_punct.str.count(r'[.!?]+')
    df.loc[df['feat_sentence_count'] == 0, 'feat_sentence_count'] = 1
    df['feat_avg_sentence_length'] = df['feat_word_count'] / df['feat_sentence_count']
    df['feat_emotion_word_count'] = texts_with_punct.apply(lambda x: len(emotion_pattern.findall(str(x))))
    df['feat_emotion_word_density'] = df['feat_emotion_word_count'] / df['feat_word_count'].replace(0, 1)
    df['feat_exclamation_count'] = texts_with_punct.str.count(r'!')
    df['feat_question_count'] = texts_with_punct.str.count(r'\?')
    return df

def extract_activity_features(df):
    print("Extracting Activity Behavior Features...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['author', 'timestamp'])
    df['feat_time_since_last_post_hrs'] = df.groupby('author')['timestamp'].diff().dt.total_seconds() / 3600.0
    df['feat_time_since_last_post_hrs'] = df['feat_time_since_last_post_hrs'].fillna(-1)
    df['hour'] = df['timestamp'].dt.hour
    df['is_late_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    print("Calculating rolling late-night ratio...")
    df['feat_late_night_ratio'] = df.groupby('author')['is_late_night'].expanding().mean().reset_index(0, drop=True)
    df['feat_is_burst'] = ((df['feat_time_since_last_post_hrs'] > 0) & (df['feat_time_since_last_post_hrs'] < 1.0)).astype(int)
    df = df.drop(columns=['hour', 'is_late_night'])
    return df

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} posts after filtering.")
    df = extract_linguistic_features(df)
    df = extract_activity_features(df)
    feature_cols = ['author', 'timestamp'] + [col for col in df.columns if col.startswith('feat_')]
    print(f"Saving {len(feature_cols)} features to {OUTPUT_FEATURES_FILE}...")
    df[feature_cols].to_csv(OUTPUT_FEATURES_FILE, index=False)
    print("Done Linguistic & Activity!")

if __name__ == "__main__":
    main()
