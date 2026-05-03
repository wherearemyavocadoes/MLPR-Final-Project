# V1 + V2 Models

## Overview

This folder contains the **final pipeline** for crisis detection. Two models are trained in sequence:

- **V1 (User-Level):** A sanity check — "Can we identify crisis users at all?"
- **V2 (Temporal):** The real task — "Can we detect a crisis BEFORE it happens?"

Both models use the **same 80-20 user-level split** (no user appears in both train and validation).

---

## V1: User-Level Classification

### What it does
Takes **all** of a user's posting history and predicts: **will this user ever enter crisis?**

### How it works
1. `build_user_dataset.py` aggregates all temporal windows into **one row per user**
2. Features are collapsed using **mean, std, min, max** across windows
3. New behavioral features are added:
   - **Posts per week** — how often the user posts
   - **Average time gap** — hours between posts
   - **Late-night ratio** — fraction of posts between 12am–5am
4. Label: if user ever posted in r/SuicideWatch → **label = 1**
5. XGBoost classifier trained on 80% of users, evaluated on 20%

### Key settings
- **No undersampling** — data is already near-balanced (48% crisis / 52% non-crisis)
- **scale_pos_weight = 1.08** — natural class ratio (nearly equal)
- **Threshold = 0.35** — optimized on validation set (best F1)

### Results

| Metric | Value |
|--------|-------|
| **Recall** | **0.876** |
| Precision | 0.743 |
| F1 Score | 0.804 |
| AUROC | 0.887 |
| Caught | 3,804 / 4,341 crisis users |
| Missed | 537 crisis users |

**Verdict:** ✅ Features contain strong signal. Proceed to V2.

---

## V2: Temporal Early Detection

### What it does
Looks at a user's **recent window of posts** and predicts: **is this user about to enter crisis?**

### How it works
1. Uses window-level data from `modeling_ready.csv` (multiple windows per user)
2. **Removes all post-crisis windows** — we can only use data from BEFORE the crisis
3. Applies the **same user split** as V1 (from `user_split.pkl`)
4. XGBoost trained on training users' windows

### Key settings
- **scale_pos_weight = 37.6** — window-level positive rate is only 2.59%
- **No undersampling** — class weighting handles the imbalance without discarding data
- **Threshold = 0.06** — much lower than default to prioritize catching crises

### Why threshold = 0.06?
The positive rate is only ~3%. The default 0.50 threshold means the model needs 50% confidence to flag a crisis, but crises are rare. At 0.50, the model only caught 8% of crisis windows. Lowering to 0.06 catches 71.4%.

In crisis detection, **missing someone in crisis (false negative) is far worse than a false alarm (false positive)**. A false alarm just sends a care message. A missed crisis could mean someone doesn't get help.

### Results

| Metric | Value |
|--------|-------|
| **Recall** | **0.714** |
| Precision | 0.118 |
| F1 Score | 0.202 |
| AUROC | 0.884 |
| Caught | 187 / 262 pre-crisis windows |
| Missed | 75 pre-crisis windows |

---

## V1 vs V2 Comparison

| | V1 (User-Level) | V2 (Temporal) |
|--|--|--|
| Question | "Is this a crisis user?" | "Is this user approaching crisis now?" |
| Data | 1 row per user | 1 row per time window |
| Positive rate | 48.2% | 2.59% |
| Difficulty | Easier (balanced data) | Harder (extreme imbalance) |
| Recall | 0.876 | 0.714 |
| AUROC | 0.887 | 0.884 |

The AUROC is nearly identical (0.887 vs 0.884), meaning **both models rank crisis cases equally well**. The difference in recall/precision is due to the different imbalance levels and thresholds.

---

## Files

| File | Description |
|------|-------------|
| `build_user_dataset.py` | Aggregates windows → 1 row per user (45,053 users, 444 features) |
| `train_v1.py` | V1 training (user-level, no undersampling) |
| `train_v2.py` | V2 training (temporal, class weighting) |
| `tune_threshold.py` | Finds optimal thresholds for both models |
| `final_evaluation.py` | Final evaluation with optimized thresholds |
| `v1_training.log` | V1 training output and metrics |
| `v2_training.log` | V2 training output and metrics |
| `threshold_tuning.log` | Full threshold sweep results |
| `final_evaluation.log` | Side-by-side comparison (default vs optimized) |
| `v1_feature_importance.csv` | Top features for V1 |
| `v2_feature_importance.csv` | Top features for V2 |

---

## How to Run

```bash
# Step 1: Build user-level dataset (requires modeling_ready.csv files)
python build_user_dataset.py

# Step 2: Train V1 (creates user_split.pkl for V2)
python train_v1.py

# Step 3: Train V2 (reuses V1's user split)
python train_v2.py

# Step 4: Optimize thresholds
python tune_threshold.py

# Step 5: Final evaluation
python final_evaluation.py
```

All results are saved to log files automatically.