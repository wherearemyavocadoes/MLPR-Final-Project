"""
Train V2: Temporal Early Detection Model
==========================================
- Reuses the SAME user-level split from V1 (user_split.pkl)
- Loads window-level data from all 4 years
- Removes post-crisis windows (for early detection)
- Trains XGBoost on temporal windows
- Evaluates with focus on Recall

IMPORTANT: Only run this AFTER v1 passes the checkpoint.
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

sys.stdout = Logger(os.path.join(V1_DIR, "v2_training.log"))


def main():
    print("=" * 60)
    print("V2: TEMPORAL EARLY DETECTION MODEL")
    print("=" * 60)

    # ============================================================
    # 1. LOAD V1 USER SPLIT
    # ============================================================
    print("\nLoading user split from V1...")
    split_data = joblib.load(os.path.join(V1_DIR, "user_split.pkl"))
    train_users_set = set(split_data['train_users'])
    val_users_set = set(split_data['val_users'])
    print(f"Train users: {len(train_users_set):,}")
    print(f"Val users:   {len(val_users_set):,}")

    # Verify no overlap
    overlap = train_users_set & val_users_set
    print(f"User overlap (should be 0): {len(overlap)}")
    assert len(overlap) == 0, "USER LEAKAGE DETECTED!"

    # ============================================================
    # 2. LOAD WINDOW-LEVEL DATA (ALL 4 YEARS)
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
    print(f"\nCombined: {len(df):,} windows, {df['author'].nunique():,} users")

    # ============================================================
    # 3. REMOVE POST-CRISIS WINDOWS (Early Detection)
    # ============================================================
    print("\n" + "=" * 60)
    print("REMOVING POST-CRISIS WINDOWS")
    print("=" * 60)

    before = len(df)

    # For crisis users: keep only windows BEFORE crisis (days_to_crisis > 0)
    # For non-crisis users: days_to_crisis is NaN, keep all
    is_crisis_user = df['is_crisis_user'] == 1
    is_pre_crisis = df['days_to_crisis'] > 0
    is_non_crisis = df['is_crisis_user'] == 0

    # Keep: non-crisis users (all windows) + crisis users (only pre-crisis windows)
    df = df[is_non_crisis | (is_crisis_user & is_pre_crisis)].copy()

    after = len(df)
    print(f"Windows before: {before:,}")
    print(f"Windows after:  {after:,}")
    print(f"Removed:        {before - after:,} post-crisis windows")

    # ============================================================
    # 4. SPLIT BY SAVED USER LISTS
    # ============================================================
    print("\n" + "=" * 60)
    print("APPLYING V1 USER SPLIT")
    print("=" * 60)

    # Define features (same as existing pipeline)
    EXCLUDE_COLS = [
        'author', 'window_start_time', 'window_end_time',
        'label', 'is_crisis_user', 'days_to_crisis', 'year',
    ]
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    X = df[feature_cols]
    y = df['label']

    # Handle NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split by user
    train_mask = df['author'].isin(train_users_set)
    val_mask = df['author'].isin(val_users_set)

    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_val = X[val_mask].copy()
    y_val = y[val_mask].copy()

    print(f"Train windows: {len(X_train):,} (positive: {y_train.sum():,}, rate: {y_train.mean()*100:.2f}%)")
    print(f"Val windows:   {len(X_val):,} (positive: {y_val.sum():,}, rate: {y_val.mean()*100:.2f}%)")

    # Users not in split (shouldn't happen but check)
    unmatched = df[~train_mask & ~val_mask]
    if len(unmatched) > 0:
        print(f"Warning: {unmatched['author'].nunique()} users not in either split. Excluding.")

    # ============================================================
    # 5. TRAIN XGBOOST (V2)
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST (V2 - Temporal)")
    print("=" * 60)

    # Class weighting for the temporal imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"scale_pos_weight: {spw:.1f}")

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
    model.fit(X_train, y_train, verbose=False)
    print("Model trained.")

    # ============================================================
    # 6. EVALUATE ON VALIDATION SET
    # ============================================================
    print("\n" + "=" * 60)
    print("V2 VALIDATION SET EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred)
    auprc = average_precision_score(y_val, y_proba)

    # AUROC can fail if only one class in val
    try:
        auroc = roc_auc_score(y_val, y_proba)
    except ValueError:
        auroc = float('nan')

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'Recall (PRIMARY)':<25} {recall:>10.4f}")
    print(f"{'Precision':<25} {precision:>10.4f}")
    print(f"{'F1 Score':<25} {f1:>10.4f}")
    print(f"{'AUPRC':<25} {auprc:>10.4f}")
    print(f"{'AUROC':<25} {auroc:>10.4f}")

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(f"  TN={cm[0,0]:>6,}   FP={cm[0,1]:>6,}")
    print(f"  FN={cm[1,0]:>6,}   TP={cm[1,1]:>6,}")
    print(f"\n  (FN = missed pre-crisis windows — minimize for early detection)")

    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Normal', 'Pre-Crisis']))

    # ============================================================
    # 7. FEATURE IMPORTANCE
    # ============================================================
    print("\n" + "=" * 60)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 60)

    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in feat_imp.head(20).iterrows():
        print(f"  {row['feature']:<45} {row['importance']:.4f}")

    feat_imp.to_csv(os.path.join(V1_DIR, "v2_feature_importance.csv"), index=False)

    # ============================================================
    # 8. SAVE MODEL
    # ============================================================
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    joblib.dump(model, os.path.join(V1_DIR, "v2_xgb_model.pkl"))
    joblib.dump(feature_cols, os.path.join(V1_DIR, "v2_feature_columns.pkl"))

    metrics = {
        'recall': recall, 'precision': precision, 'f1': f1,
        'auroc': auroc, 'auprc': auprc,
        'train_windows': len(X_train), 'val_windows': len(X_val),
        'train_positive': int(y_train.sum()), 'val_positive': int(y_val.sum()),
        'scale_pos_weight': spw,
    }
    joblib.dump(metrics, os.path.join(V1_DIR, "v2_metrics.pkl"))

    print("Saved: v2_xgb_model.pkl")
    print("Saved: v2_feature_columns.pkl")
    print("Saved: v2_feature_importance.csv")
    print("Saved: v2_metrics.pkl")

    # ============================================================
    # 9. FINAL ASSESSMENT
    # ============================================================
    print("\n" + "=" * 60)
    print("V2 FINAL ASSESSMENT")
    print("=" * 60)
    print(f"Recall (early detection):  {recall:.4f}")
    print(f"AUPRC:                     {auprc:.4f}")
    if recall >= 0.50:
        print("✅ V2 achieves reasonable recall for early detection.")
    else:
        print("⚠️  V2 recall is below 0.50. May need further tuning.")

    print("\n✅ V2 COMPLETE.")


if __name__ == "__main__":
    main()
