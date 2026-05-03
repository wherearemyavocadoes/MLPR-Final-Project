import os
import glob
import pandas as pd

# 1. Configuration
INPUT_DIR = "/Users/arya_vachhani/Downloads/Reddit Data/raw data copy/2021"
OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2021_user_temporal.csv"

def load_data(input_dir):
    """Load all 2021 CSV files into a single DataFrame."""
    all_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
    df_list = []
    
    print(f"Found {len(all_files)} CSV files in {input_dir}.")
    for f in all_files:
        try:
            temp_df = pd.read_csv(f, low_memory=False, lineterminator='\n')
            df_list.append(temp_df)
        except Exception as e:
            try:
                temp_df = pd.read_csv(f, low_memory=False)
                df_list.append(temp_df)
            except Exception as e2:
                print(f"Failed to read {f}: {e2}")
                
    if not df_list:
        raise ValueError("No data could be loaded.")
        
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(full_df)} total rows.")
    return full_df

def clean_data(df):
    """
    1. Basic Cleaning
    - Remove null posts
    - Remove deleted/removed content
    - Standardize timestamps
    - Ensure user IDs exist
    """
    initial_rows = len(df)
    
    df = df[df['author'].notna()]
    df = df[~df['author'].isin(['[deleted]', '[removed]'])]
    
    if 'selftext' in df.columns:
        df['selftext'] = df['selftext'].fillna('')
    else:
        df['selftext'] = ''
        
    if 'title' in df.columns:
        df['title'] = df['title'].fillna('')
    else:
        df['title'] = ''
        
    if 'body' in df.columns:
        df['body'] = df['body'].fillna('')
    else:
        df['body'] = ''
        
    df['combined_text'] = df['title'] + " " + df['selftext'] + " " + df['body']
    df['combined_text'] = df['combined_text'].str.strip()
    
    df = df[df['combined_text'] != '']
    
    deleted_markers = ['[deleted]', '[removed]', '[deleted] [deleted]', '[removed] [removed]']
    df = df[~df['combined_text'].isin(deleted_markers)]
    
    if 'selftext' in df.columns:
        df = df[~df['selftext'].isin(['[deleted]', '[removed]'])]
        
    if 'created_utc' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
    df = df[df['timestamp'].notna()]
    
    print(f"Cleaned data: {initial_rows} rows -> {len(df)} rows.")
    return df

def group_and_sort(df):
    """
    2. User-Level Grouping
    - Group posts by: user_id (author)
    - Then sort each user's posts by: timestamp (ascending)
    """
    print("Sorting data by author and timestamp...")
    df_sorted = df.sort_values(by=['author', 'timestamp'], ascending=[True, True])
    return df_sorted

def main():
    df = load_data(INPUT_DIR)
    
    print("Starting basic cleaning...")
    df_clean = clean_data(df)
    
    print("Starting user-level grouping and sorting...")
    df_final = group_and_sort(df_clean)
    
    print(f"Saving final processed data ({len(df_final)} rows) to {OUTPUT_FILE}...")
    if 'combined_text' in df_final.columns:
        df_final = df_final.drop(columns=['combined_text'])
        
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
