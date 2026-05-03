# Preprocessing Pipeline

## Overview

The preprocessing pipeline converts raw Reddit posts into structured, modeling-ready data. It runs in **5 stages**, applied to each year of data (2019–2022) separately.

**Input:** Raw Reddit posts from mental health subreddits
**Output:** `processed_{year}_modeling_ready.csv` — temporal windows with 400+ features per window

---

## Stages

### Stage 1 — Data Cleaning

- Loads raw Reddit data (posts from r/depression, r/anxiety, r/mentalhealth, r/lonely, r/SuicideWatch)
- Removes deleted/removed posts
- Filters to users with ≥5 posts (ensures enough data per user)
- Sorts by user and timestamp
- **Output:** `processed_{year}_user_temporal.csv`

### Stage 2 — Text Processing

- Creates two text streams:
  - **Title + selftext** (for submissions)
  - **Body** (for comments)
- Combines into a single clean text field
- Basic text cleaning (lowercase, whitespace normalization)
- **Output:** `processed_{year}_text_dual.csv`

### Stage 3 — Feature Extraction (3 sub-stages + merge)

#### 3a. Semantic Features
- Generates **384-dimensional text embeddings** using a Transformer model (sentence-transformers)
- Each post gets a dense vector capturing its meaning
- **Output:** Embedding columns (`emb_000` to `emb_383`)

#### 3b. Linguistic Features
- Word count, sentence count
- Posting frequency
- Activity patterns (time of day, day of week)
- **Output:** Linguistic feature columns

#### 3c. Psychological Features
- **Anxiety score** — presence of anxiety-related language
- **Sadness score** — depression-related language
- **Negative emotion score** — general negative sentiment
- **Output:** Psychological feature columns

#### 3d. Merge
- Combines semantic + linguistic + psychological features into one dataset
- **Output:** `processed_{year}_features.csv`

### Stage 4 — Temporal Windowing

- Groups each user's posts into **sliding time windows**
- Window size: **10 posts**
- Stride: **5 posts** (50% overlap)
- Each window gets aggregated features (mean of embeddings, mean of linguistic/psychological scores)
- Adds window-level metadata: posting frequency, time span, post count
- **Output:** `processed_{year}_temporal_windows.csv`

### Stage 5 — Crisis Labeling

- Identifies each user's **first SuicideWatch post** as the crisis point
- Labels windows relative to this point:
  - **Pre-crisis (label=1):** Last 3 windows before the crisis
  - **Normal (label=0):** All other windows
  - **Post-crisis:** Windows after crisis (removed during modeling)
- Adds `days_to_crisis` column for temporal analysis
- **Output:** `processed_{year}_modeling_ready.csv`

---

## Output Schema

Each `modeling_ready.csv` file contains:

| Column Group | Count | Description |
|-------------|-------|-------------|
| `win_emb_000` to `win_emb_383` | 384 | Transformer embedding features (averaged per window) |
| `win_mean_anxiety`, `win_mean_sadness`, etc. | ~15 | Linguistic and psychological features |
| `post_count`, `win_posting_freq_per_day` | ~5 | Window metadata |
| `author` | 1 | User identifier |
| `label` | 1 | 0 = normal, 1 = pre-crisis |
| `is_crisis_user` | 1 | Whether user ever posted in SuicideWatch |
| `days_to_crisis` | 1 | Days until first SuicideWatch post |

**Total:** ~404 columns per window

---

## Data Volume

| Year | Windows | Users |
|------|---------|-------|
| 2019 | 17,478 | — |
| 2020 | 22,297 | — |
| 2021 | 28,049 | — |
| 2022 | 16,363 | — |
| **Total** | **84,187** | **45,053** |

---

## How to Run

Run stages in order (1 → 2 → 3 → 4 → 5) for each year:

```bash
# Example for 2019
python "stage 1_2019.py"
python "stage 2_2019.py"
python "stage 3_semantic_2019.py"
python "stage 3_linguistics_2019.py"
python "stage 3_psychological_2019.py"
python "stage 3_merge_2019.py"
python "stage 4_2019.py"
python "stage 5_2019.py"
```

Repeat for 2020, 2021, and 2022.