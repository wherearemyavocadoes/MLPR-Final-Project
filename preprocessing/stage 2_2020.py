import pandas as pd
import re
import emoji
from tqdm import tqdm
import multiprocessing as mp
import os

INPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2020_user_temporal.csv"
OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2020_text_dual.csv"

# Regex for URLs:
url_pattern = re.compile(r'https?://\S+|www\.\S+')
# Regex for special characters: Keep only alphanumeric and spaces
special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')

def remove_urls(text):
    if not isinstance(text, str):
        return ""
    return url_pattern.sub('', text)

def stream_a_cleaning(text):
    """
    Stream A: Raw Text (for embeddings)
    - Lowercase
    - Remove URLs
    - Light cleaning only
    """
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower()
    text = remove_urls(text)
    # light cleaning: normalize whitespace
    text = " ".join(text.split())
    return text

def stream_b_cleaning(text):
    """
    Stream B: Clean Text (for LIWC + behavioral metrics)
    - Lowercase
    - Remove URLs
    - Remove emojis
    - Remove special characters
    """
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower()
    text = remove_urls(text)
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    # Remove special characters (keep only alphanumeric and spaces)
    text = special_char_pattern.sub(' ', text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text

def process_chunk_a(texts):
    return [stream_a_cleaning(t) for t in texts]

def process_chunk_b(texts):
    return [stream_b_cleaning(t) for t in texts]

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")

    print("Combining text fields...")
    df['title'] = df['title'].fillna('')
    df['selftext'] = df['selftext'].fillna('')
    if 'body' in df.columns:
        df['body'] = df['body'].fillna('')
        df['combined_text'] = df['title'] + " " + df['selftext'] + " " + df['body']
    else:
        df['combined_text'] = df['title'] + " " + df['selftext']
        
    df['combined_text'] = df['combined_text'].apply(lambda x: x.strip())
    texts_list = df['combined_text'].tolist()

    num_cores = max(1, os.cpu_count() - 1)
    print(f"Using {num_cores} cores for multiprocessing.")
    chunk_size = len(texts_list) // num_cores + 1
    chunks = [texts_list[i:i + chunk_size] for i in range(0, len(texts_list), chunk_size)]

    print("Processing Stream A (Raw Text for embeddings)...")
    clean_a = []
    with mp.Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap(process_chunk_a, chunks), total=len(chunks)):
            clean_a.extend(result)
    df['text_raw'] = clean_a

    print("Processing Stream B (Clean Text for LIWC)...")
    clean_b = []
    with mp.Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap(process_chunk_b, chunks), total=len(chunks)):
            clean_b.extend(result)
    df['text_clean'] = clean_b

    # Drop intermediate combined_text
    df = df.drop(columns=['combined_text'])
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    # Required for safe multiprocessing on macOS
    mp.set_start_method('spawn', force=True)
    main()
