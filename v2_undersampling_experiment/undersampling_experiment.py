"""
V2 Undersampling Experiment (85/15)
=====================================
Tests whether undersampling the majority class to 85% neg / 15% pos
performs better than the current approach (class weighting + threshold tuning).

- Uses the SAME user split from V1 (user_split.pkl)
- Same data pipeline as current V2 (remove post-crisis windows)
- Only difference: undersampling training data to 85/15 ratio
- Compares results side-by-side with current V2

This is an EXPERIMENT — does NOT modify any existing files.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
)
import joblib
import os
import sys
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = "/Users/arya_vachhani/Downloads/Reddit Data"
V1_DIR = os.path.join(BASE_DIR, "v1_user_level_model")
EXP_DIR = os.path.join(BASE_DIR, "v2_undersampling_experiment")
YEARS = [2019, 2020, 2021, 2022]

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

sys.stdout = Logger(os.path.join(EXP_DIR, "undersampling_experiment.log"))


def undersample_to_ratio(X_train, y_train, target_pos_ratio=0.15):
    """Undersample negative class to achieve target positive ratio."""
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    print(f"\nBefore undersampling: {n_neg:,} negative, {n_pos:,} positive ({n_pos/(n_pos+n_neg)*100:.2f}%)")

    # target: pos / total = target_pos_ratio
    # Keep all positives, reduce negatives
    target_n_neg = int(n_pos * (1 - target_pos_ratio) / target_pos_ratio)

    neg_indices = y_train[y_train == 0].index
    pos_indices = y_train[y_train == 1].index

    sampled_neg = np.random.RandomState(42).choice(neg_indices, size=target_n_neg, replace=False)
    keep_indices = np.concatenate([pos_indices.values, sampled_neg])

    X_out = X_train.loc[keep_indices].copy()
    y_out = y_train.loc[keep_indices].copy()

    n_pos_after = (y_out == 1).sum()
    n_neg_after = (y_out == 0).sum()
    print(f"Undersampled NEGATIVE: {n_neg:,} → {n_neg_after:,}")
    print(f"After: {n_neg_after:,} negative, {n_pos_after:,} positive ({n_pos_after/(n_pos_after+n_neg_after)*100:.1f}%)")
    print(f"Data removed: {n_neg - n_neg_after:,} negative windows ({(n_neg - n_neg_after)/n_neg*100:.1f}%)")

    return X_out, y_out


def find_best_threshold(y_true, y_proba):
    """Sweep thresholds, return results df and best-F1 row."""
    results = []
    for t in np.arange(0.01, 0.99, 0.005):
        y_pred = (y_proba >= t).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        r = recall_score(y_true, y_pred)
        p = precision_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred)
        results.append({'threshold': t, 'recall': r, 'precision': p, 'f1': f})
    df = pd.DataFrame(results)
    best_f1 = df.loc[df['f1'].idxmax()]
    return df, best_f1


def evaluate_at_threshold(y_true, y_proba, threshold, label):
    """Full evaluation at a given threshold."""
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
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pre-Crisis']))

    return {'threshold': threshold, 'recall': recall, 'precision': precision,
            'f1': f1, 'auroc': auroc, 'auprc': auprc,
            'TN': int(cm[0,0]), 'FP': int(cm[0,1]),
            'FN': int(cm[1,0]), 'TP': int(cm[1,1])}


def main():
    print("=" * 60)
    print("V2 UNDERSAMPLING EXPERIMENT (85/15)")
    print("=" * 60)

    # ============================================================
    # 1. LOAD USER SPLIT FROM V1
    # ============================================================
    print("\nLoading V1 user split...")
    split_data = joblib.load(os.path.join(V1_DIR, "user_split.pkl"))
    train_users_set = set(split_data['train_users'])
    val_users_set = set(split_data['val_users'])
    print(f"Train users: {len(train_users_set):,}, Val users: {len(val_users_set):,}")

    # ============================================================
    # 2. LOAD WINDOW-LEVEL DATA
    # ============================================================
    print("\n" + "=" * 60)
    print("LOADING WINDOW-LEVEL DATA")
    print("=" * 60)

    frames = []
    for year in YEARS:
        path = os.path.join(BASE_DIR, f"processed_{year}_modeling_ready.csv")
        print(f"Loading {year}...", end=" ", flush=True)
        df = pd.read_csv(path)
        df['year'] = year
        print(f"{len(df):,} windows")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(df):,} windows")

    # ============================================================
    # 3. REMOVE POST-CRISIS WINDOWS
    # ============================================================
    print("\n" + "=" * 60)
    print("REMOVING POST-CRISIS WINDOWS")
    print("=" * 60)

    before = len(df)
    is_crisis = df['is_crisis_user'] == 1
    is_pre = df['days_to_crisis'] > 0
    is_non = df['is_crisis_user'] == 0
    df = df[is_non | (is_crisis & is_pre)].copy()
    print(f"Before: {before:,} → After: {len(df):,} (removed {before - len(df):,})")

    # ============================================================
    # 4. PREPARE FEATURES AND SPLIT
    # ============================================================
    EXCLUDE = ['author', 'window_start_time', 'window_end_time',
               'label', 'is_crisis_user', 'days_to_crisis', 'year']
    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['label']

    train_mask = df['author'].isin(train_users_set)
    val_mask = df['author'].isin(val_users_set)

    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_val = X[val_mask].copy()
    y_val = y[val_mask].copy()

    print(f"\nTrain: {len(X_train):,} windows ({y_train.sum():,} positive, {y_train.mean()*100:.2f}%)")
    print(f"Val:   {len(X_val):,} windows ({y_val.sum():,} positive, {y_val.mean()*100:.2f}%)")

    # ============================================================
    # 5. UNDERSAMPLE TRAINING SET TO 85/15
    # ============================================================
    print("\n" + "=" * 60)
    print("UNDERSAMPLING TRAINING SET (85% neg, 15% pos)")
    print("=" * 60)

    X_train_us, y_train_us = undersample_to_ratio(X_train, y_train, target_pos_ratio=0.15)

    # ============================================================
    # 6. TRAIN XGBOOST (with undersampled data)
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST (Undersampled 85/15)")
    print("=" * 60)

    # scale_pos_weight on the undersampled data
    n_neg_us = (y_train_us == 0).sum()
    n_pos_us = (y_train_us == 1).sum()
    spw = n_neg_us / n_pos_us
    print(f"scale_pos_weight: {spw:.2f}")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        tree_method='hist',
        scale_pos_weight=spw,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=400,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.0,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )

    print("Training...")
    model.fit(X_train_us, y_train_us, verbose=False)
    print("Model trained.")

    # ============================================================
    # 7. EVALUATE ON FULL VALIDATION SET
    # ============================================================
    y_proba = model.predict_proba(X_val)[:, 1]

    # Default threshold
    print("\n" + "=" * 60)
    print("EVALUATION AT DEFAULT THRESHOLD (0.50)")
    print("=" * 60)
    evaluate_at_threshold(y_val, y_proba, 0.50, 'default')

    # Find optimal threshold
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 60)

    results_df, best_f1 = find_best_threshold(y_val, y_proba)
    print(f"Best F1 threshold: {best_f1['threshold']:.4f} (F1={best_f1['f1']:.4f})")

    # High recall thresholds
    for target_recall in [0.50, 0.70]:
        hr = results_df[results_df['recall'] >= target_recall]
        if len(hr) > 0:
            best_hr = hr.loc[hr['precision'].idxmax()]
            print(f"Best at Recall≥{target_recall:.2f}: threshold={best_hr['threshold']:.4f}, "
                  f"R={best_hr['recall']:.4f}, P={best_hr['precision']:.4f}, F1={best_hr['f1']:.4f}")

    # Evaluate at best F1
    print("\n" + "=" * 60)
    print(f"EVALUATION AT BEST-F1 THRESHOLD ({best_f1['threshold']:.2f})")
    print("=" * 60)
    us_metrics = evaluate_at_threshold(y_val, y_proba, best_f1['threshold'], 'best_f1')

    # Evaluate at recall≥0.70 threshold
    hr70 = results_df[results_df['recall'] >= 0.70]
    if len(hr70) > 0:
        best_hr70 = hr70.loc[hr70['precision'].idxmax()]
        print("\n" + "=" * 60)
        print(f"EVALUATION AT RECALL≥0.70 THRESHOLD ({best_hr70['threshold']:.2f})")
        print("=" * 60)
        us_metrics_hr = evaluate_at_threshold(y_val, y_proba, best_hr70['threshold'], 'recall_70')

    # Threshold sweep
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP")
    print("=" * 60)
    print(f"{'Threshold':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print("-" * 42)
    for t in [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        row = results_df.iloc[(results_df['threshold'] - t).abs().argsort()[:1]].iloc[0]
        print(f"{row['threshold']:>10.3f} {row['recall']:>10.4f} {row['precision']:>10.4f} {row['f1']:>10.4f}")

    # ============================================================
    # 8. SIDE-BY-SIDE COMPARISON
    # ============================================================
    print("\n\n" + "=" * 60)
    print("COMPARISON: CURRENT V2 vs UNDERSAMPLED V2")
    print("=" * 60)

    # Load current V2 metrics
    v2_current = joblib.load(os.path.join(V1_DIR, "v2_final_metrics.pkl"))

    print(f"\n{'Metric':<20} {'Current V2':>15} {'Undersampled':>15} {'Winner':>12}")
    print(f"{'':20} {'(weight+thresh)':>15} {'(85/15)':>15}")
    print("-" * 65)

    comparisons = [
        ('Threshold', v2_current['threshold'], best_f1['threshold']),
        ('Recall', v2_current['recall'], us_metrics['recall']),
        ('Precision', v2_current['precision'], us_metrics['precision']),
        ('F1 Score', v2_current['f1'], us_metrics['f1']),
        ('AUROC', v2_current['auroc'], us_metrics['auroc']),
        ('AUPRC', v2_current['auprc'], us_metrics['auprc']),
        ('Missed (FN)', v2_current['FN'], us_metrics['FN']),
        ('Caught (TP)', v2_current['TP'], us_metrics['TP']),
    ]

    for name, curr, us in comparisons:
        if name in ['Missed (FN)']:
            winner = "Current" if curr < us else "Undersamp" if us < curr else "Tie"
        elif name in ['Caught (TP)', 'Recall', 'Precision', 'F1 Score', 'AUROC', 'AUPRC']:
            winner = "Current" if curr > us else "Undersamp" if us > curr else "Tie"
        else:
            winner = ""

        if isinstance(curr, float) and name != 'Threshold':
            print(f"  {name:<18} {curr:>15.4f} {us:>15.4f} {winner:>12}")
        elif name == 'Threshold':
            print(f"  {name:<18} {curr:>15.2f} {us:>15.2f} {winner:>12}")
        else:
            print(f"  {name:<18} {curr:>15,} {us:>15,} {winner:>12}")

    # Also compare at matched recall level (~0.70)
    if len(hr70) > 0:
        print(f"\n--- At matched recall ≥0.70 ---")
        print(f"  Current V2:      Recall={v2_current['recall']:.4f}, Precision={v2_current['precision']:.4f} (threshold={v2_current['threshold']:.2f})")
        print(f"  Undersampled V2: Recall={us_metrics_hr['recall']:.4f}, Precision={us_metrics_hr['precision']:.4f} (threshold={best_hr70['threshold']:.2f})")

    # ============================================================
    # 9. SAVE ARTIFACTS
    # ============================================================
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)

    joblib.dump(model, os.path.join(EXP_DIR, "v2_undersampled_model.pkl"))
    joblib.dump(us_metrics, os.path.join(EXP_DIR, "v2_undersampled_metrics.pkl"))
    joblib.dump(feature_cols, os.path.join(EXP_DIR, "v2_undersampled_features.pkl"))

    print("Saved: v2_undersampled_model.pkl")
    print("Saved: v2_undersampled_metrics.pkl")
    print("Saved: undersampling_experiment.log")
    print("\n✅ UNDERSAMPLING EXPERIMENT COMPLETE.")


if __name__ == "__main__":
    main()
