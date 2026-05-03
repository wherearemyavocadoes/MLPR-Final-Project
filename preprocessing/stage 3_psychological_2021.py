import pandas as pd
import re
from tqdm import tqdm

INPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_filtered.csv"
OUTPUT_FEATURES_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/stage3_features_psychological_2021.csv"

LIWC_DICT = {
    'anxiety': ['worried', 'nervous', 'anxious', 'scared', 'afraid', 'panic', 'terror', 'dread', 'fear', 'tension', 'stress'],
    'sadness': ['crying', 'cry', 'grief', 'sad', 'sadness', 'depressed', 'depressing', 'sorrow', 'tears', 'heartbreak', 'gloomy', 'miserable', 'loss', 'hopeless'],
    'anger': ['hate', 'kill', 'annoyed', 'angry', 'mad', 'furious', 'rage', 'pissed', 'frustrated', 'resent', 'bitter', 'screw', 'hate'],
    'negative_emotion': ['bad', 'weird', 'awful', 'terrible', 'worst', 'hurt', 'pain', 'disgust', 'nasty', 'gross', 'ugly', 'sick', 'guilt', 'shame', 'embarrassed', 'fail', 'stupid', 'dumb'],
    'positive_emotion': ['love', 'nice', 'sweet', 'good', 'great', 'happy', 'happier', 'joy', 'excited', 'proud', 'smile', 'laugh', 'funny', 'awesome', 'amazing', 'perfect'],
    'first_person': ['i', 'me', 'my', 'mine', 'myself'],
    'cognitive_process': ['cause', 'know', 'think', 'believe', 'maybe', 'perhaps', 'wonder', 'realize', 'understand', 'figure', 'decide', 'guess', 'suppose', 'reason', 'why']
}

compiled_patterns = {}
for category, words in LIWC_DICT.items():
    pattern_string = r'\b(?:' + '|'.join(words) + r')\b'
    compiled_patterns[category] = re.compile(pattern_string, re.IGNORECASE)

def extract_psychological_features(df):
    print("Extracting Psychological Features (LIWC approximation)...")
    texts = df['text_clean'].fillna("")
    word_counts = texts.str.count(' ') + 1
    word_counts = word_counts.replace(0, 1)
    tqdm.pandas()
    for category, pattern in compiled_patterns.items():
        print(f"  -> Counting {category} words...")
        raw_count = texts.progress_apply(lambda x: len(pattern.findall(str(x))))
        df[f'feat_psych_{category}'] = raw_count / word_counts
    return df

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} posts.")
    df = extract_psychological_features(df)
    feature_cols = ['author', 'timestamp'] + [col for col in df.columns if col.startswith('feat_')]
    print(f"Saving {len(feature_cols)} features to {OUTPUT_FEATURES_FILE}...")
    df[feature_cols].to_csv(OUTPUT_FEATURES_FILE, index=False)
    print("Done Psychological extraction!")

if __name__ == "__main__":
    main()
