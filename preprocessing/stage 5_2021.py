import pandas as pd
import numpy as np
from tqdm import tqdm

BASE_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_features.csv"
WINDOWS_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_temporal_windows.csv"
OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_modeling_ready.csv"

CRISIS_SUBREDDIT = "SuicideWatch"
PRE_CRISIS_WINDOWS = 3

def determine_crisis_timestamps(df_base):
    print("Precomputing crisis timestamps per user...")
    df_base['subreddit'] = df_base['subreddit'].str.lower()
    crisis_posts = df_base[df_base['subreddit'] == 'suicidewatch']
    crisis_posts = crisis_posts.sort_values(['author', 'timestamp'])
    first_crisis = crisis_posts.drop_duplicates(subset=['author'], keep='first')
    first_crisis['timestamp'] = pd.to_datetime(first_crisis['timestamp'])
    crisis_dict = dict(zip(first_crisis['author'], first_crisis['timestamp']))
    print(f"Detected {len(crisis_dict)} users with a crisis point.")
    return crisis_dict

def assign_labels(df_windows, crisis_dict):
    print("Assigning Target Labels to temporal windows...")
    df_windows['window_end_time'] = pd.to_datetime(df_windows['window_end_time'])
    df_windows['label'] = 0
    df_windows['is_crisis_user'] = 0
    df_windows['days_to_crisis'] = np.nan
    labeled_indices = []
    user_groups = df_windows.groupby('author')
    for author, group in tqdm(user_groups, total=len(user_groups)):
        if author not in crisis_dict:
            continue
        crisis_time = crisis_dict[author]
        df_windows.loc[group.index, 'is_crisis_user'] = 1
        df_windows.loc[group.index, 'days_to_crisis'] = (crisis_time - group['window_end_time']).dt.total_seconds() / 86400.0
        pre_crisis_mask = group['window_end_time'] < crisis_time
        pre_crisis_windows = group[pre_crisis_mask]
        if len(pre_crisis_windows) == 0:
            continue
        pre_crisis_windows = pre_crisis_windows.sort_values('window_end_time')
        target_indices = pre_crisis_windows.tail(PRE_CRISIS_WINDOWS).index.tolist()
        labeled_indices.extend(target_indices)
    df_windows.loc[labeled_indices, 'label'] = 1
    print(f"\nLabel Distribution across ALL windows:")
    print(df_windows['label'].value_counts())
    print("\nLabeled Windows Summary:")
    print(f"Total positive (Class 1) windows: {len(labeled_indices)}")
    return df_windows

def main():
    print("Loading Base File for Ground Truth tracking...")
    df_base = pd.read_csv(BASE_FILE, usecols=['author', 'timestamp', 'subreddit'])
    crisis_dict = determine_crisis_timestamps(df_base)
    del df_base
    print(f"\nLoading Temporal Windows File ({WINDOWS_FILE})...")
    df_windows = pd.read_csv(WINDOWS_FILE)
    df_labeled = assign_labels(df_windows, crisis_dict)
    print(f"\nSaving finalized predictive dataset: {OUTPUT_FILE}")
    df_labeled.to_csv(OUTPUT_FILE, index=False)
    print("Stage 5 Complete!")

if __name__ == "__main__":
    main()
