import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from scipy.spatial.distance import cosine

INPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_features.csv"
OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_temporal_windows.csv"

WINDOW_SIZE = 10
STRIDE = 5
EMBED_DIM = 384

def calculate_window_features(df_user):
    df_user = df_user.sort_values('timestamp').reset_index(drop=True)
    num_posts = len(df_user)
    if num_posts < WINDOW_SIZE:
        pass
    windows = []
    for start_idx in range(0, num_posts, STRIDE):
        end_idx = min(start_idx + WINDOW_SIZE, num_posts)
        if end_idx - start_idx < 2:
            break
        window_df = df_user.iloc[start_idx:end_idx]
        mean_anxiety = window_df['feat_psych_anxiety'].mean()
        mean_sadness = window_df['feat_psych_sadness'].mean()
        mean_negative_emotion = window_df['feat_psych_negative_emotion'].mean()
        emb_cols = [f"feat_emb_{i:03d}" for i in range(EMBED_DIM)]
        mean_embedding = window_df[emb_cols].mean(axis=0).values
        time_span_days = (pd.to_datetime(window_df['timestamp'].iloc[-1]) - pd.to_datetime(window_df['timestamp'].iloc[0])).total_seconds() / 86400.0
        if time_span_days < 1.0:
            time_span_days = 1.0
        posting_freq = len(window_df) / time_span_days
        hour_series = pd.to_datetime(window_df['timestamp']).dt.hour
        late_night_count = ((hour_series >= 0) & (hour_series <= 5)).sum()
        night_ratio = late_night_count / len(window_df)
        volatility_sadness = window_df['feat_psych_sadness'].var(ddof=1)
        volatility_anxiety = window_df['feat_psych_anxiety'].var(ddof=1)
        if pd.isna(volatility_sadness): volatility_sadness = 0.0
        if pd.isna(volatility_anxiety): volatility_anxiety = 0.0
        windows.append({
            'author': df_user['author'].iloc[0],
            'window_start_time': window_df['timestamp'].iloc[0],
            'window_end_time': window_df['timestamp'].iloc[-1],
            'post_count': len(window_df),
            'win_mean_anxiety': mean_anxiety,
            'win_mean_sadness': mean_sadness,
            'win_mean_negative_emotion': mean_negative_emotion,
            'win_posting_freq_per_day': posting_freq,
            'win_night_ratio': night_ratio,
            'win_volatility_sadness': volatility_sadness,
            'win_volatility_anxiety': volatility_anxiety,
            'mean_embedding': mean_embedding
        })
        if end_idx == num_posts:
            break
    final_windows = []
    for i in range(len(windows)):
        win = windows[i]
        if i == 0:
            win['delta_anxiety'] = 0.0
            win['delta_sadness'] = 0.0
            win['delta_negative_emotion'] = 0.0
            win['delta_posting_freq'] = 0.0
            win['delta_night_ratio'] = 0.0
            win['embedding_drift'] = 0.0
        else:
            prev_win = windows[i-1]
            win['delta_anxiety'] = win['win_mean_anxiety'] - prev_win['win_mean_anxiety']
            win['delta_sadness'] = win['win_mean_sadness'] - prev_win['win_mean_sadness']
            win['delta_negative_emotion'] = win['win_mean_negative_emotion'] - prev_win['win_mean_negative_emotion']
            win['delta_posting_freq'] = win['win_posting_freq_per_day'] - prev_win['win_posting_freq_per_day']
            win['delta_night_ratio'] = win['win_night_ratio'] - prev_win['win_night_ratio']
            if np.all(win['mean_embedding'] == 0) or np.all(prev_win['mean_embedding'] == 0):
                drift = 0.0
            else:
                drift = cosine(win['mean_embedding'], prev_win['mean_embedding'])
            win['embedding_drift'] = drift
        for d in range(EMBED_DIM):
            win[f'win_emb_{d:03d}'] = win['mean_embedding'][d]
        final_windows.append(win)
    for win in final_windows:
        if 'mean_embedding' in win:
            del win['mean_embedding']
    return final_windows

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} posts. Grouping by user to construct windows...")
    all_temporal_windows = []
    author_groups = df.groupby('author')
    for author, group_df in tqdm(author_groups, total=len(author_groups)):
        user_windows = calculate_window_features(group_df)
        all_temporal_windows.extend(user_windows)
    print(f"\nGenerated {len(all_temporal_windows)} total temporal windows across {len(author_groups)} users.")
    print("Converting to DataFrame...")
    df_temporal = pd.DataFrame(all_temporal_windows)
    print(f"Saving temporal features to {OUTPUT_FILE}...")
    df_temporal.to_csv(OUTPUT_FILE, index=False)
    print("Done Stage 4!")

if __name__ == "__main__":
    main()
