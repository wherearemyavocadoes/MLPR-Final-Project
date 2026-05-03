"""
Train V1: User-Level Crisis Detection Model
=============================================
- Loads user-level dataset (one row per user)
- 80-20 stratified user-level split
- Undersamples majority class to 85% neg / 15% pos
- Trains XGBoost with simple hyperparameters
- Evaluates: Recall, Precision, F1, AUPRC, AUROC, Confusion Matrix

This is a MANDATORY CHECKPOINT: v2 only proceeds if v1 recall is reasonable.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
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

BASE_DIR = "/Users/arya_vachhani/Downloads/Reddit Data/v1_user_level_model"
INPUT_FILE = os.path.join(BASE_DIR, "user_level_dataset.csv")

# Logging to file + console
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

sys.stdout = Logger(os.path.join(BASE_DIR, "v1_training.log"))


def undersample_to_ratio(X_train, y_train, target_pos_ratio=0.15):
    """
    Undersample to achieve target_pos_ratio (positive as fraction of total).
    If positive class is already above target, undersample positives.
    If negative class is majority and positive is minority, undersample negatives.
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    print(f"\nBefore undersampling: {n_neg:,} negative, {n_pos:,} positive ({n_pos/(n_pos+n_neg)*100:.1f}% positive)")

    # Target: pos / (pos + neg) = target_pos_ratio
    # If pos rate > target: undersample positives, keep all negatives
    # If pos rate < target: undersample negatives, keep all positives
    current_pos_ratio = n_pos / (n_pos + n_neg)

    if current_pos_ratio > target_pos_ratio:
        # Undersample POSITIVE class
        target_n_pos = int(n_neg * target_pos_ratio / (1 - target_pos_ratio))
        pos_indices = y_train[y_train == 1].index
        neg_indices = y_train[y_train == 0].index
        sampled_pos = np.random.RandomState(42).choice(pos_indices, size=target_n_pos, replace=False)
        keep_indices = np.concatenate([neg_indices.values, sampled_pos])
        print(f"Undersampled POSITIVE class: {n_pos:,} → {target_n_pos:,}")
    else:
        # Undersample NEGATIVE class
        target_n_neg = int(n_pos * (1 - target_pos_ratio) / target_pos_ratio)
        pos_indices = y_train[y_train == 1].index
        neg_indices = y_train[y_train == 0].index
        sampled_neg = np.random.RandomState(42).choice(neg_indices, size=target_n_neg, replace=False)
        keep_indices = np.concatenate([pos_indices.values, sampled_neg])
        print(f"Undersampled NEGATIVE class: {n_neg:,} → {target_n_neg:,}")

    X_out = X_train.loc[keep_indices].copy()
    y_out = y_train.loc[keep_indices].copy()

    n_pos_after = (y_out == 1).sum()
    n_neg_after = (y_out == 0).sum()
    print(f"After undersampling: {n_neg_after:,} negative, {n_pos_after:,} positive ({n_pos_after/(n_pos_after+n_neg_after)*100:.1f}% positive)")

    return X_out, y_out


def main():
    print("=" * 60)
    print("V1: USER-LEVEL CRISIS DETECTION MODEL")
    print("=" * 60)

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("\nLoading user-level dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total users: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # Separate features and label
    EXCLUDE_COLS = ['author', 'label']
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"Feature columns: {len(feature_cols)}")

    X = df[feature_cols]
    y = df['label']

    # Handle NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\nLabel distribution:")
    print(f"  Non-crisis (0): {(y == 0).sum():,}")
    print(f"  Crisis (1):     {(y == 1).sum():,}")
    print(f"  Crisis rate:    {y.mean() * 100:.1f}%")

    # ============================================================
    # 2. USER-LEVEL 80-20 SPLIT
    # ============================================================
    print("\n" + "=" * 60)
    print("USER-LEVEL TRAIN/VALIDATION SPLIT (80-20)")
    print("=" * 60)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save user split for v2 reuse
    train_users = df.loc[X_train.index, 'author'].tolist()
    val_users = df.loc[X_val.index, 'author'].tolist()

    split_data = {
        'train_users': train_users,
        'val_users': val_users,
    }
    joblib.dump(split_data, os.path.join(BASE_DIR, "user_split.pkl"))

    print(f"Train users: {len(train_users):,}")
    print(f"Val users:   {len(val_users):,}")
    print(f"Train crisis: {(y_train == 1).sum():,} ({y_train.mean()*100:.1f}%)")
    print(f"Val crisis:   {(y_val == 1).sum():,} ({y_val.mean()*100:.1f}%)")

    # Verify no user leakage
    overlap = set(train_users) & set(val_users)
    print(f"User overlap (should be 0): {len(overlap)}")
    assert len(overlap) == 0, "USER LEAKAGE DETECTED!"

    print(f"\nSaved user split: user_split.pkl")

    # ============================================================
    # 3. NO UNDERSAMPLING (data is near-balanced at 48/52)
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING DATA (no undersampling — near-balanced)")
    print("=" * 60)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    print(f"Training on ALL data: {n_neg:,} negative, {n_pos:,} positive ({n_pos/(n_pos+n_neg)*100:.1f}%)")

    # ============================================================
    # 4. TRAIN XGBOOST (simple parameters)
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST (V1 - Simple, No Undersampling)")
    print("=" * 60)

    # scale_pos_weight from actual class ratio (near 1.0 since balanced)
    spw = n_neg / n_pos
    print(f"scale_pos_weight: {spw:.2f}")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        tree_method='hist',
        scale_pos_weight=spw,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
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
    # 5. EVALUATE ON FULL VALIDATION SET
    # ============================================================
    print("\n" + "=" * 60)
    print("V1 VALIDATION SET EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auroc = roc_auc_score(y_val, y_proba)
    auprc = average_precision_score(y_val, y_proba)

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'Recall (PRIMARY)':<25} {recall:>10.4f}")
    print(f"{'Precision':<25} {precision:>10.4f}")
    print(f"{'F1 Score':<25} {f1:>10.4f}")
    print(f"{'AUROC':<25} {auroc:>10.4f}")
    print(f"{'AUPRC':<25} {auprc:>10.4f}")

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(f"  TN={cm[0,0]:>6,}   FP={cm[0,1]:>6,}")
    print(f"  FN={cm[1,0]:>6,}   TP={cm[1,1]:>6,}")
    print(f"\n  (FN = missed crisis users — should be as low as possible)")

    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Non-Crisis', 'Crisis']))

    # ============================================================
    # 6. FEATURE IMPORTANCE (Top 20)
    # ============================================================
    print("\n" + "=" * 60)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 60)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for _, row in feat_imp.head(20).iterrows():
        print(f"  {row['feature']:<45} {row['importance']:.4f}")

    feat_imp.to_csv(os.path.join(BASE_DIR, "v1_feature_importance.csv"), index=False)

    # ============================================================
    # 7. SAVE MODEL AND ARTIFACTS
    # ============================================================
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    joblib.dump(model, os.path.join(BASE_DIR, "v1_xgb_model.pkl"))
    joblib.dump(feature_cols, os.path.join(BASE_DIR, "v1_feature_columns.pkl"))

    metrics = {
        'recall': recall, 'precision': precision, 'f1': f1,
        'auroc': auroc, 'auprc': auprc,
        'train_users': len(train_users), 'val_users': len(val_users),
        'train_crisis': int((y_train == 1).sum()),
        'val_crisis': int((y_val == 1).sum()),
    }
    joblib.dump(metrics, os.path.join(BASE_DIR, "v1_metrics.pkl"))

    print("Saved: v1_xgb_model.pkl")
    print("Saved: v1_feature_columns.pkl")
    print("Saved: v1_feature_importance.csv")
    print("Saved: v1_metrics.pkl")
    print("Saved: user_split.pkl")

    # ============================================================
    # 8. CHECKPOINT DECISION
    # ============================================================
    print("\n" + "=" * 60)
    print("V1 CHECKPOINT")
    print("=" * 60)

    if recall >= 0.50:
        print(f"✅ PASS — Recall = {recall:.4f} (≥ 0.50)")
        print("→ V1 successfully detects crisis users. Proceed to V2.")
    elif recall >= 0.30:
        print(f"⚠️  MARGINAL — Recall = {recall:.4f} (0.30–0.50)")
        print("→ V1 shows some signal. Consider proceeding to V2 with caution.")
    else:
        print(f"❌ FAIL — Recall = {recall:.4f} (< 0.30)")
        print("→ V1 cannot reliably detect crisis users. DO NOT proceed to V2.")
        print("→ Revisit feature engineering or data pipeline.")


if __name__ == "__main__":
    main()
