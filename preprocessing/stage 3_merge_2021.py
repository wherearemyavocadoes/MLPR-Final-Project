import pandas as pd
import numpy as np
import os
import glob
import pickle
from tqdm import tqdm

INPUT_BASE_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_filtered.csv"
LINGUISTIC_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/stage3_features_linguistic_activity_2021.csv"
PSYCHOLOGICAL_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/stage3_features_psychological_2021.csv"
EMBEDDINGS_DIR = "/Users/arya_vachhani/Downloads/Reddit Data/embeddings_2021"
OUTPUT_MERGED_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_features.csv"

def merge_features():
    print("Loading base filtered dataset...")
    df_base = pd.read_csv(INPUT_BASE_FILE)
    print("Loading Linguistic & Activity Features...")
    df_ling = pd.read_csv(LINGUISTIC_FILE)
    print("Loading Psychological Features...")
    df_psych = pd.read_csv(PSYCHOLOGICAL_FILE)
    assert len(df_base) == len(df_ling) == len(df_psych), "Row counts do not match!"
    drop_cols = ['author', 'timestamp']
    df_ling_clean = df_ling.drop(columns=drop_cols)
    df_psych_clean = df_psych.drop(columns=drop_cols)
    print("Concatenating tabular features...")
    df_merged = pd.concat([df_base, df_ling_clean, df_psych_clean], axis=1)
    print(f"Loading {len(glob.glob(os.path.join(EMBEDDINGS_DIR, '*.pkl')))} batch embedding files...")
    num_posts = len(df_base)
    embed_dim = 384
    embeddings_matrix = np.zeros((num_posts, embed_dim), dtype=np.float32)
    batch_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "batch_*.pkl"))
    for file in tqdm(batch_files, desc="Unrolling embeddings"):
        with open(file, 'rb') as f:
            batch_data = pickle.load(f)
            indices = batch_data['indices']
            emb_values = batch_data['embeddings']
            embeddings_matrix[indices] = emb_values
    print("Converting embeddings to DataFrame columns (this may take a moment)...")
    emb_cols = [f"feat_emb_{i:03d}" for i in range(embed_dim)]
    df_embeddings = pd.DataFrame(embeddings_matrix, columns=emb_cols)
    print("Final concatenation...")
    df_final = pd.concat([df_merged, df_embeddings], axis=1)
    print(f"Saving final dataset with shape {df_final.shape} to {OUTPUT_MERGED_FILE}...")
    df_final.to_csv(OUTPUT_MERGED_FILE, index=False)
    print("Done! Stage 3 Complete.")

if __name__ == "__main__":
    merge_features()
