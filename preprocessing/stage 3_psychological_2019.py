import pandas as pd
import re
from tqdm import tqdm

INPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_filtered.csv"
OUTPUT_FEATURES_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/stage3_features_psychological.csv"

# LIWC Approximation Dictionary (Based on core well-known terms for these categories)
# Using \b to match whole words and | as OR separator. We use Stream B (clean text) which has no punctuation.
LIWC_DICT = {
    'anxiety': ['worried', 'nervous', 'anxious', 'scared', 'afraid', 'panic', 'terror', 'dread', 'fear', 'tension', 'stress'],
    'sadness': ['crying', 'cry', 'grief', 'sad', 'sadness', 'depressed', 'depressing', 'sorrow', 'tears', 'heartbreak', 'gloomy', 'miserable', 'loss', 'hopeless'],
    'anger': ['hate', 'kill', 'annoyed', 'angry', 'mad', 'furious', 'rage', 'pissed', 'frustrated', 'resent', 'bitter', 'screw', 'hate'],
    'negative_emotion': ['bad', 'weird', 'awful', 'terrible', 'worst', 'hurt', 'pain', 'disgust', 'nasty', 'gross', 'ugly', 'sick', 'guilt', 'shame', 'embarrassed', 'fail', 'stupid', 'dumb'],
    'positive_emotion': ['love', 'nice', 'sweet', 'good', 'great', 'happy', 'happier', 'joy', 'excited', 'proud', 'smile', 'laugh', 'funny', 'awesome', 'amazing', 'perfect'],
    'first_person': ['i', 'me', 'my', 'mine', 'myself'],
    'cognitive_process': ['cause', 'know', 'think', 'believe', 'maybe', 'perhaps', 'wonder', 'realize', 'understand', 'figure', 'decide', 'guess', 'suppose', 'reason', 'why']
}

# Compile regex patterns
compiled_patterns = {}
for category, words in LIWC_DICT.items():
    # We add (?:) for non-capturing group, \b for word boundaries
    pattern_string = r'\b(?:' + '|'.join(words) + r')\b'
    compiled_patterns[category] = re.compile(pattern_string, re.IGNORECASE)

def extract_psychological_features(df):
    """
    2. Psychological Features (open-source LIWC proxy)
    - Anxiety
    - Sadness
    - Anger
    - Negative emotion
    - Positive emotion
    - First-person pronouns
    - Cognitive process words
    """
    print("Extracting Psychological Features (LIWC approximation)...")
    
    # We use Stream B text which already has punctuation and emojis stripped
    texts = df['text_clean'].fillna("")
    
    # For normalization, we need the word count of Stream B
    # Since punctuation is gone, we can just split by space
    word_counts = texts.str.count(' ') + 1
    word_counts = word_counts.replace(0, 1) # Prevent div by zero
    
    tqdm.pandas()
    for category, pattern in compiled_patterns.items():
        print(f"  -> Counting {category} words...")
        # Find all matches and get count
        raw_count = texts.progress_apply(lambda x: len(pattern.findall(str(x))))
        # Provide density (LIWC standard is usually percentage of total words)
        df[f'feat_psych_{category}'] = raw_count / word_counts
        
    return df

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} posts.")
    
    # Extract Psychological
    df = extract_psychological_features(df)
    
    feature_cols = ['author', 'timestamp'] + [col for col in df.columns if col.startswith('feat_')]
    
    print(f"Saving {len(feature_cols)} features to {OUTPUT_FEATURES_FILE}...")
    df[feature_cols].to_csv(OUTPUT_FEATURES_FILE, index=False)
    print("Done Psychological extraction!")

if __name__ == "__main__":
    main()
