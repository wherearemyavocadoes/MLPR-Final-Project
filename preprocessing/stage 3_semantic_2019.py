import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import pickle

# Configuration
INPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_text_dual.csv"
OUTPUT_EMBEDDINGS_DIR = "/Users/arya_vachhani/Downloads/Reddit Data/embeddings_2019"
FILTERED_OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_filtered.csv"
BATCH_SIZE = 500  # Safe for standard laptops
MIN_POSTS = 5 # --> bound to change

os.makedirs(OUTPUT_EMBEDDINGS_DIR, exist_ok=True)

def filter_users(df):
    """Filter out users with < MIN_POSTS posts."""
    print(f"Original dataset size: {len(df)} rows, {df['author'].nunique()} users.")
    user_counts = df['author'].value_counts()
    valid_users = user_counts[user_counts >= MIN_POSTS].index
    
    df_filtered = df[df['author'].isin(valid_users)].copy()
    print(f"Filtered dataset size: {len(df_filtered)} rows, {df_filtered['author'].nunique()} users.")
    return df_filtered

def extract_semantic_features(df):
    """
    1. Semantic Features (Sentence Transformer Embeddings)
    - all-MiniLM-L6-v2 
    - 384-dimensional embedding per post
    - Batch processing to save memory
    """
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    # This model is very fast and lightweight but accurate for sentence semantics
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # We use Stream A (text_raw) for embeddings
    texts = df['text_raw'].fillna("").tolist()
    total_posts = len(texts)
    
    print(f"Starting batch embedding generation ({total_posts} total posts)...")
    
    # Create an empty list to store the mapping of index to batch file
    # We won't store the actual embeddings in memory to prevent RAM overflow
    batch_files = []
    
    for i in tqdm(range(0, total_posts, BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_indices = list(range(i, min(i + BATCH_SIZE, total_posts)))
        
        # Encode batch
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        
        # Save batch to disk immediately
        batch_filename = os.path.join(OUTPUT_EMBEDDINGS_DIR, f"batch_{i}.pkl")
        with open(batch_filename, 'wb') as f:
            pickle.dump({
                'indices': batch_indices,
                'embeddings': batch_embeddings
            }, f)
            
        batch_files.append(batch_filename)
        
    print(f"Successfully generated and saved {len(batch_files)} batches of embeddings to {OUTPUT_EMBEDDINGS_DIR}.")

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Filtering users with >= {MIN_POSTS} posts...")
    df_filtered = filter_users(df)
    
    # Save the filtered dataset so subsequent feature extractions use the same cohort
    # Reset index so that our batch embedding indices align perfectly with the CSV rows
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered.to_csv(FILTERED_OUTPUT_FILE, index=False)
    print(f"Saved filtered cohort to {FILTERED_OUTPUT_FILE}.")
    
    # Run Semantic Feature Extraction
    extract_semantic_features(df_filtered)

if __name__ == "__main__":
    main()
