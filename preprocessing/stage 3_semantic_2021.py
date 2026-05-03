import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import pickle

INPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_text_dual.csv"
OUTPUT_EMBEDDINGS_DIR = "/Users/arya_vachhani/Downloads/Reddit Data/embeddings_2021"
FILTERED_OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_filtered.csv"
BATCH_SIZE = 500
MIN_POSTS = 5

os.makedirs(OUTPUT_EMBEDDINGS_DIR, exist_ok=True)

def filter_users(df):
    print(f"Original dataset size: {len(df)} rows, {df['author'].nunique()} users.")
    user_counts = df['author'].value_counts()
    valid_users = user_counts[user_counts >= MIN_POSTS].index
    df_filtered = df[df['author'].isin(valid_users)].copy()
    print(f"Filtered dataset size: {len(df_filtered)} rows, {df_filtered['author'].nunique()} users.")
    return df_filtered

def extract_semantic_features(df):
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = df['text_raw'].fillna("").tolist()
    total_posts = len(texts)
    print(f"Starting batch embedding generation ({total_posts} total posts)...")
    batch_files = []
    for i in tqdm(range(0, total_posts, BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_indices = list(range(i, min(i + BATCH_SIZE, total_posts)))
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        batch_filename = os.path.join(OUTPUT_EMBEDDINGS_DIR, f"batch_{i}.pkl")
        with open(batch_filename, 'wb') as f:
            pickle.dump({'indices': batch_indices, 'embeddings': batch_embeddings}, f)
        batch_files.append(batch_filename)
    print(f"Successfully generated and saved {len(batch_files)} batches of embeddings to {OUTPUT_EMBEDDINGS_DIR}.")

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Filtering users with >= {MIN_POSTS} posts...")
    df_filtered = filter_users(df)
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered.to_csv(FILTERED_OUTPUT_FILE, index=False)
    print(f"Saved filtered cohort to {FILTERED_OUTPUT_FILE}.")
    extract_semantic_features(df_filtered)

if __name__ == "__main__":
    main()
