"""
Build User-Level Dataset (V1)
=============================
Aggregates window-level features from ALL 4 years (2019-2022)
into a single user-level dataset (one row per user).

Input:  processed_{year}_modeling_ready.csv (for each year)
Output: v1_user_level_model/user_level_dataset.csv
"""

import pandas as pd
import numpy as np
import os
import time

BASE_DIR = "/Users/arya_vachhani/Downloads/Reddit Data"
OUTPUT_DIR = os.path.join(BASE_DIR, "v1_user_level_model")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "user_level_dataset.csv")

YEARS = [2019, 2020, 2021, 2022]

# Feature columns to aggregate (excluding metadata and embeddings)
BEHAVIORAL_FEATURES = [
    'post_count',
    'win_mean_anxiety', 'win_mean_sadness', 'win_mean_negative_emotion',
    'win_posting_freq_per_day', 'win_night_ratio',
    'win_volatility_sadness', 'win_volatility_anxiety',
    'delta_anxiety', 'delta_sadness', 'delta_negative_emotion',
    'delta_posting_freq', 'delta_night_ratio', 'embedding_drift',
]

EMBED_COLS = [f'win_emb_{i:03d}' for i in range(384)]

METADATA_COLS = [
    'author', 'window_start_time', 'window_end_time',
    'label', 'is_crisis_user', 'days_to_crisis', 'year'
]


def load_all_years():
    """Load and concatenate modeling_ready data from all years."""
    frames = []
    for year in YEARS:
        path = os.path.join(BASE_DIR, f"processed_{year}_modeling_ready.csv")
        print(f"Loading {year}...", end=" ", flush=True)
        df = pd.read_csv(path)
        df['year'] = year
        print(f"{len(df):,} windows, {df['author'].nunique():,} users")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined):,} windows, {combined['author'].nunique():,} unique users")
    return combined


def aggregate_user_features(df):
    """Aggregate window-level features to user-level (one row per user)."""
    print("\nAggregating features to user level...")
    start = time.time()

    # --- 1. Determine user-level label ---
    user_label = df.groupby('author')['is_crisis_user'].max().reset_index()
    user_label.columns = ['author', 'label']

    # --- 2. Build aggregation dictionary for behavioral features ---
    agg_dict = {}
    for feat in BEHAVIORAL_FEATURES:
        if feat == 'post_count':
            # Sum gives total posts; also take mean/std
            agg_dict[feat] = ['sum', 'mean', 'std']
        else:
            agg_dict[feat] = ['mean', 'std', 'min', 'max']

    # --- 3. Aggregate embeddings (mean only to keep dims manageable) ---
    for col in EMBED_COLS:
        agg_dict[col] = ['mean']

    # --- 4. Run the aggregation ---
    user_agg = df.groupby('author').agg(agg_dict)

    # Flatten multi-level column names
    user_agg.columns = ['_'.join(col).strip() for col in user_agg.columns]

    # Rename embedding columns (remove _mean suffix for cleanliness)
    rename_map = {}
    for col in EMBED_COLS:
        rename_map[f'{col}_mean'] = f'user_{col}'
    user_agg.rename(columns=rename_map, inplace=True)

    # Rename post_count_sum to total_posts
    user_agg.rename(columns={'post_count_sum': 'total_posts'}, inplace=True)

    user_agg = user_agg.reset_index()

    # --- 5. Compute new behavioral features ---
    print("Computing behavioral features...")

    # Number of windows per user
    window_counts = df.groupby('author').size().reset_index(name='num_windows')
    user_agg = user_agg.merge(window_counts, on='author')

    # Time range: earliest window_start to latest window_end
    df['window_start_time'] = pd.to_datetime(df['window_start_time'])
    df['window_end_time'] = pd.to_datetime(df['window_end_time'])

    time_range = df.groupby('author').agg(
        first_post=('window_start_time', 'min'),
        last_post=('window_end_time', 'max')
    ).reset_index()

    time_range['date_range_days'] = (
        time_range['last_post'] - time_range['first_post']
    ).dt.total_seconds() / 86400.0
    time_range['date_range_days'] = time_range['date_range_days'].clip(lower=1.0)

    user_agg = user_agg.merge(time_range[['author', 'date_range_days']], on='author')

    # Posts per week
    user_agg['posts_per_week'] = user_agg['total_posts'] / (user_agg['date_range_days'] / 7.0)

    # Average time gap between windows (proxy for time gap between posts)
    def compute_avg_gap(group):
        times = group['window_end_time'].sort_values()
        if len(times) < 2:
            return 0.0
        gaps = times.diff().dropna().dt.total_seconds() / 3600.0  # hours
        return gaps.mean()

    avg_gaps = df.groupby('author').apply(compute_avg_gap).reset_index(name='avg_time_gap_hrs')
    user_agg = user_agg.merge(avg_gaps, on='author')

    # Late-night ratio (user-level mean of window-level night ratio)
    # Already captured by win_night_ratio_mean, but add explicit column
    user_agg['late_night_ratio'] = user_agg['win_night_ratio_mean']

    # --- 6. Merge with labels ---
    user_agg = user_agg.merge(user_label, on='author')

    elapsed = time.time() - start
    print(f"Aggregation complete in {elapsed:.1f}s")

    return user_agg


def main():
    print("=" * 60)
    print("BUILD USER-LEVEL DATASET (V1)")
    print("=" * 60)

    # Load data
    df = load_all_years()

    # Aggregate
    user_df = aggregate_user_features(df)

    # Handle NaN/inf
    feature_cols = [c for c in user_df.columns if c not in ['author', 'label']]
    user_df[feature_cols] = user_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    nan_cols = user_df[feature_cols].isnull().sum()
    if nan_cols.sum() > 0:
        print(f"\nWarning: {(nan_cols > 0).sum()} columns have NaN values. Filling with 0.")
        user_df[feature_cols] = user_df[feature_cols].fillna(0)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total users: {len(user_df):,}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Crisis users (label=1): {(user_df['label'] == 1).sum():,}")
    print(f"Non-crisis users (label=0): {(user_df['label'] == 0).sum():,}")
    print(f"Crisis rate: {user_df['label'].mean() * 100:.1f}%")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    user_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")
    print("✅ User-level dataset built successfully.")


if __name__ == "__main__":
    main()
