"""
Threshold Tuning for V1 and V2 Models
=======================================
- Loads saved models and recreates validation sets
- Sweeps thresholds on validation set to find optimal
- Reports metrics at: best-F1 threshold, best-recall@precision>=X, and custom thresholds
- Re-evaluates both models with optimized thresholds
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve,
)
import joblib
import os
import sys
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = "/Users/arya_vachhani/Downloads/Reddit Data"
V1_DIR = os.path.join(BASE_DIR, "v1_user_level_model")

# Logging
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass

sys.stdout = Logger(os.path.join(V1_DIR, "threshold_tuning.log"))


def find_best_threshold(y_true, y_proba, metric='f1'):
    """Sweep thresholds and find the one that maximizes the given metric."""
    thresholds = np.arange(0.01, 0.99, 0.005)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        r = recall_score(y_true, y_pred)
        p = precision_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred)
        results.append({'threshold': t, 'recall': r, 'precision': p, 'f1': f})

    df = pd.DataFrame(results)

    if metric == 'f1':
        best = df.loc[df['f1'].idxmax()]
    elif metric == 'recall':
        best = df.loc[df['recall'].idxmax()]
    else:
        best = df.loc[df['f1'].idxmax()]

    return df, best


def evaluate_at_threshold(y_true, y_proba, threshold, label=''):
    """Evaluate model at a specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'Threshold':<25} {threshold:>10.4f}")
    print(f"{'Recall (PRIMARY)':<25} {recall:>10.4f}")
    print(f"{'Precision':<25} {precision:>10.4f}")
    print(f"{'F1 Score':<25} {f1:>10.4f}")
    print(f"{'AUROC':<25} {auroc:>10.4f}")
    print(f"{'AUPRC':<25} {auprc:>10.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:>6,}   FP={cm[0,1]:>6,}")
    print(f"  FN={cm[1,0]:>6,}   TP={cm[1,1]:>6,}")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
          target_names=['Non-Crisis', 'Crisis'] if 'V1' in label else ['Normal', 'Pre-Crisis']))

    return {'threshold': threshold, 'recall': recall, 'precision': precision,
            'f1': f1, 'auroc': auroc, 'auprc': auprc}


def tune_v1():
    """Tune threshold for V1 user-level model."""
    print("=" * 60)
    print("V1 THRESHOLD TUNING (User-Level Model)")
    print("=" * 60)

    # Load model and data
    model = joblib.load(os.path.join(V1_DIR, "v1_xgb_model.pkl"))
    feature_cols = joblib.load(os.path.join(V1_DIR, "v1_feature_columns.pkl"))
    split_data = joblib.load(os.path.join(V1_DIR, "user_split.pkl"))

    df = pd.read_csv(os.path.join(V1_DIR, "user_level_dataset.csv"))
    val_users = set(split_data['val_users'])

    val_mask = df['author'].isin(val_users)
    X_val = df.loc[val_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = df.loc[val_mask, 'label']

    print(f"Val users: {len(val_users):,}")
    print(f"Val crisis: {(y_val == 1).sum():,} / {len(y_val):,}")

    # Get probabilities
    y_proba = model.predict_proba(X_val)[:, 1]

    # --- Default threshold (0.5) ---
    print("\n" + "-" * 60)
    print("DEFAULT THRESHOLD (0.50)")
    print("-" * 60)
    evaluate_at_threshold(y_val, y_proba, 0.50, label='V1')

    # --- Find best F1 threshold ---
    print("\n" + "-" * 60)
    print("SEARCHING FOR OPTIMAL THRESHOLDS...")
    print("-" * 60)

    results_df, best_f1 = find_best_threshold(y_val, y_proba, metric='f1')

    print(f"\nBest F1 threshold: {best_f1['threshold']:.4f}")
    print(f"  → Recall={best_f1['recall']:.4f}, Precision={best_f1['precision']:.4f}, F1={best_f1['f1']:.4f}")

    # --- Find threshold for high recall (≥0.80) with best precision ---
    high_recall = results_df[results_df['recall'] >= 0.80]
    if len(high_recall) > 0:
        best_hr = high_recall.loc[high_recall['precision'].idxmax()]
        print(f"\nBest threshold at Recall≥0.80: {best_hr['threshold']:.4f}")
        print(f"  → Recall={best_hr['recall']:.4f}, Precision={best_hr['precision']:.4f}, F1={best_hr['f1']:.4f}")

    # --- Find threshold for very high recall (≥0.90) ---
    very_high_recall = results_df[results_df['recall'] >= 0.90]
    if len(very_high_recall) > 0:
        best_vhr = very_high_recall.loc[very_high_recall['precision'].idxmax()]
        print(f"\nBest threshold at Recall≥0.90: {best_vhr['threshold']:.4f}")
        print(f"  → Recall={best_vhr['recall']:.4f}, Precision={best_vhr['precision']:.4f}, F1={best_vhr['f1']:.4f}")

    # --- Evaluate at best F1 threshold ---
    print("\n" + "=" * 60)
    print(f"V1 RE-EVALUATION AT BEST-F1 THRESHOLD ({best_f1['threshold']:.4f})")
    print("=" * 60)
    evaluate_at_threshold(y_val, y_proba, best_f1['threshold'], label='V1')

    # --- Evaluate at high-recall threshold ---
    if len(high_recall) > 0:
        print("\n" + "=" * 60)
        print(f"V1 RE-EVALUATION AT HIGH-RECALL THRESHOLD ({best_hr['threshold']:.4f})")
        print("=" * 60)
        evaluate_at_threshold(y_val, y_proba, best_hr['threshold'], label='V1')

    # --- Show threshold sweep summary ---
    print("\n" + "=" * 60)
    print("V1 THRESHOLD SWEEP (sampled)")
    print("=" * 60)
    print(f"{'Threshold':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print("-" * 42)
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
        row = results_df.iloc[(results_df['threshold'] - t).abs().argsort()[:1]].iloc[0]
        print(f"{row['threshold']:>10.2f} {row['recall']:>10.4f} {row['precision']:>10.4f} {row['f1']:>10.4f}")

    return best_f1['threshold']


def tune_v2():
    """Tune threshold for V2 temporal model."""
    print("\n\n" + "=" * 60)
    print("V2 THRESHOLD TUNING (Temporal Model)")
    print("=" * 60)

    # Load model and split
    model = joblib.load(os.path.join(V1_DIR, "v2_xgb_model.pkl"))
    feature_cols = joblib.load(os.path.join(V1_DIR, "v2_feature_columns.pkl"))
    split_data = joblib.load(os.path.join(V1_DIR, "user_split.pkl"))
    val_users_set = set(split_data['val_users'])

    # Load window-level data (same as train_v2.py)
    frames = []
    for year in [2019, 2020, 2021, 2022]:
        path = os.path.join(BASE_DIR, f"processed_{year}_modeling_ready.csv")
        df = pd.read_csv(path)
        df['year'] = year
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Remove post-crisis windows
    is_crisis = df['is_crisis_user'] == 1
    is_pre = df['days_to_crisis'] > 0
    is_non = df['is_crisis_user'] == 0
    df = df[is_non | (is_crisis & is_pre)].copy()

    # Validation split
    EXCLUDE_COLS = ['author', 'window_start_time', 'window_end_time',
                    'label', 'is_crisis_user', 'days_to_crisis', 'year']
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    val_mask = df['author'].isin(val_users_set)
    X_val = df.loc[val_mask, feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = df.loc[val_mask, 'label']

    print(f"Val windows: {len(X_val):,}")
    print(f"Val positive: {(y_val == 1).sum():,} ({y_val.mean()*100:.2f}%)")

    y_proba = model.predict_proba(X_val)[:, 1]

    # --- Default threshold ---
    print("\n" + "-" * 60)
    print("DEFAULT THRESHOLD (0.50)")
    print("-" * 60)
    evaluate_at_threshold(y_val, y_proba, 0.50, label='V2')

    # --- Find best thresholds ---
    print("\n" + "-" * 60)
    print("SEARCHING FOR OPTIMAL THRESHOLDS...")
    print("-" * 60)

    results_df, best_f1 = find_best_threshold(y_val, y_proba, metric='f1')

    print(f"\nBest F1 threshold: {best_f1['threshold']:.4f}")
    print(f"  → Recall={best_f1['recall']:.4f}, Precision={best_f1['precision']:.4f}, F1={best_f1['f1']:.4f}")

    # High recall threshold
    high_recall = results_df[results_df['recall'] >= 0.50]
    if len(high_recall) > 0:
        best_hr = high_recall.loc[high_recall['precision'].idxmax()]
        print(f"\nBest threshold at Recall≥0.50: {best_hr['threshold']:.4f}")
        print(f"  → Recall={best_hr['recall']:.4f}, Precision={best_hr['precision']:.4f}, F1={best_hr['f1']:.4f}")

    very_high = results_df[results_df['recall'] >= 0.70]
    if len(very_high) > 0:
        best_vh = very_high.loc[very_high['precision'].idxmax()]
        print(f"\nBest threshold at Recall≥0.70: {best_vh['threshold']:.4f}")
        print(f"  → Recall={best_vh['recall']:.4f}, Precision={best_vh['precision']:.4f}, F1={best_vh['f1']:.4f}")

    # --- Evaluate at best F1 ---
    print("\n" + "=" * 60)
    print(f"V2 RE-EVALUATION AT BEST-F1 THRESHOLD ({best_f1['threshold']:.4f})")
    print("=" * 60)
    evaluate_at_threshold(y_val, y_proba, best_f1['threshold'], label='V2')

    # --- Evaluate at high-recall ---
    if len(high_recall) > 0:
        print("\n" + "=" * 60)
        print(f"V2 RE-EVALUATION AT RECALL≥0.50 THRESHOLD ({best_hr['threshold']:.4f})")
        print("=" * 60)
        evaluate_at_threshold(y_val, y_proba, best_hr['threshold'], label='V2')

    # --- Threshold sweep ---
    print("\n" + "=" * 60)
    print("V2 THRESHOLD SWEEP (sampled)")
    print("=" * 60)
    print(f"{'Threshold':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print("-" * 42)
    for t in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        row = results_df.iloc[(results_df['threshold'] - t).abs().argsort()[:1]].iloc[0]
        print(f"{row['threshold']:>10.3f} {row['recall']:>10.4f} {row['precision']:>10.4f} {row['f1']:>10.4f}")

    return best_f1['threshold']


def main():
    print("=" * 60)
    print("THRESHOLD TUNING — V1 & V2 MODELS")
    print("=" * 60)

    v1_best = tune_v1()
    v2_best = tune_v2()

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"V1 optimal threshold: {v1_best:.4f} (was 0.50)")
    print(f"V2 optimal threshold: {v2_best:.4f} (was 0.50)")
    print("\n✅ Threshold tuning complete.")


if __name__ == "__main__":
    main()
