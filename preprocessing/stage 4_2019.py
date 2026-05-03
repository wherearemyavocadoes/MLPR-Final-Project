import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from scipy.spatial.distance import cosine

# Configuration
INPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_features.csv"
OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_temporal_windows.csv"

WINDOW_SIZE = 10
STRIDE = 5
EMBED_DIM = 384

def calculate_window_features(df_user):
    """
    Computes aggregates and inter-window deltas for a single user's posts.
    Returns a list of dictionaries (one per window).
    """
    # Sort chronologically just in case
    df_user = df_user.sort_values('timestamp').reset_index(drop=True)
    num_posts = len(df_user)
    
    if num_posts < WINDOW_SIZE:
        # User has fewer posts than window size, skip or create 1 partial window?
        # Given dataset is already filtered to >= 5, a user with 5-9 will form one partial window.
        # But to have deltas (t vs t-1) you need at least 2 windows. 
        # Let's just create as many windows as possible using stride.
        pass
        
    windows = []
    
    # Generate windows
    for start_idx in range(0, num_posts, STRIDE):
        end_idx = min(start_idx + WINDOW_SIZE, num_posts)
        
        # If the window has fewer than 2 posts, we can't compute variance or meaningful aggregates.
        # Skip if it's the very last tiny trailing window of size 1.
        if end_idx - start_idx < 2:
            break
            
        window_df = df_user.iloc[start_idx:end_idx]
        
        # Compute Window-Level Aggregates
        mean_anxiety = window_df['feat_psych_anxiety'].mean()
        mean_sadness = window_df['feat_psych_sadness'].mean()
        mean_negative_emotion = window_df['feat_psych_negative_emotion'].mean()
        
        # Mean embedding (384-d)
        emb_cols = [f"feat_emb_{i:03d}" for i in range(EMBED_DIM)]
        mean_embedding = window_df[emb_cols].mean(axis=0).values
        
        # Posting frequency (posts per day within window)
        # Time span of the window
        time_span_days = (pd.to_datetime(window_df['timestamp'].iloc[-1]) - pd.to_datetime(window_df['timestamp'].iloc[0])).total_seconds() / 86400.0
        # If spanning basically 0 time (e.g. all in same day), frequency is high
        if time_span_days < 1.0:
            time_span_days = 1.0
        posting_freq = len(window_df) / time_span_days
        
        # Night-post ratio (We used feat_late_night_ratio which was expanding mean, but let's recalculate for just this window)
        # Or using the raw hours:
        hour_series = pd.to_datetime(window_df['timestamp']).dt.hour
        late_night_count = ((hour_series >= 0) & (hour_series <= 5)).sum()
        night_ratio = late_night_count / len(window_df)
        
        # Emotional volatility (variance of sadness/anxiety)
        # Using ddof=0 or 1. If length=1, var is NaN if ddof=1. We rely on length >= 2.
        volatility_sadness = window_df['feat_psych_sadness'].var(ddof=1)
        volatility_anxiety = window_df['feat_psych_anxiety'].var(ddof=1)
        
        # Handle nan variances if all values are identical
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
            'mean_embedding': mean_embedding # Storing as numpy array temporarily
        })
        
        # If we reached the end of the array, break
        if end_idx == num_posts:
            break
            
    # Compute Change-Based Features (Deltas) between t and t-1
    final_windows = []
    
    for i in range(len(windows)):
        win = windows[i]
        
        if i == 0:
            # First window has no prior temporal delta
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
            
            # Embedding drift = cosine distance (1 - cosine similarity)
            # if either vector is all zeros (rare), cosine distance might fail or be 1
            if np.all(win['mean_embedding'] == 0) or np.all(prev_win['mean_embedding'] == 0):
                drift = 0.0
            else:
                drift = cosine(win['mean_embedding'], prev_win['mean_embedding'])
            win['embedding_drift'] = drift
            
        # We unroll the mean embedding back into cols so it saves trivially to CSV
        for d in range(EMBED_DIM):
            win[f'win_emb_{d:03d}'] = win['mean_embedding'][d]
            
        final_windows.append(win)
        
    for win in final_windows:
        if 'mean_embedding' in win:
            del win['mean_embedding']
            
    return final_windows

def main():
    print(f"Loading {INPUT_FILE}...")
    # Because of the 400+ columns, loading can be slow, but 106k fits in memory.
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} posts. Grouping by user to construct windows...")
    
    # We use a pure list accumulation pattern. (Much faster than appending DF rows)
    all_temporal_windows = []
    
    # Group processing
    # groupby -> apply is notoriously slow in pandas for custom heavily operations.
    # But since we are generating multiple rows per group, iterative grouping is clearer
    # and we can use tqdm for progress tracking.
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
