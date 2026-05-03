"""
Final Evaluation with Optimized Thresholds
============================================
V1 threshold: 0.15 (best F1)
V2 threshold: 0.06 (recall-focused for crisis detection)

Saves final metrics, confusion matrices, and updated model artifacts.
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

V1_THRESHOLD = 0.35
V2_THRESHOLD = 0.06

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

sys.stdout = Logger(os.path.join(V1_DIR, "final_evaluation.log"))


def evaluate_and_save(y_true, y_proba, threshold, model_name, target_names):
    """Full evaluation at a given threshold. Returns metrics dict."""
    y_pred = (y_proba >= threshold).astype(int)

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'Threshold':<25} {threshold:>10.4f}")
    print(f"{'Recall (PRIMARY)':<25} {recall:>10.4f}")
    print(f"{'Precision':<25} {precision:>10.4f}")
    print(f"{'F1 Score':<25} {f1:>10.4f}")
    print(f"{'AUROC':<25} {auroc:>10.4f}")
    print(f"{'AUPRC':<25} {auprc:>10.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:>6,}   FP={cm[0,1]:>6,}")
    print(f"  FN={cm[1,0]:>6,}   TP={cm[1,1]:>6,}")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    return {
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'TN': int(cm[0, 0]),
        'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]),
        'TP': int(cm[1, 1]),
    }


def main():
    print("=" * 60)
    print("FINAL EVALUATION WITH OPTIMIZED THRESHOLDS")
    print("=" * 60)
    print(f"V1 threshold: {V1_THRESHOLD}")
    print(f"V2 threshold: {V2_THRESHOLD}")

    # Load shared resources
    split_data = joblib.load(os.path.join(V1_DIR, "user_split.pkl"))
    val_users_set = set(split_data['val_users'])

    # ==================================================================
    # V1 FINAL EVALUATION
    # ==================================================================
    print("\n\n" + "=" * 60)
    print(f"V1 FINAL RESULTS (threshold={V1_THRESHOLD})")
    print("=" * 60)

    v1_model = joblib.load(os.path.join(V1_DIR, "v1_xgb_model.pkl"))
    v1_features = joblib.load(os.path.join(V1_DIR, "v1_feature_columns.pkl"))

    df_user = pd.read_csv(os.path.join(V1_DIR, "user_level_dataset.csv"))
    val_mask = df_user['author'].isin(val_users_set)
    X_val_v1 = df_user.loc[val_mask, v1_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val_v1 = df_user.loc[val_mask, 'label']

    print(f"Validation users: {val_mask.sum():,}")
    print(f"Crisis users in val: {(y_val_v1 == 1).sum():,}")

    y_proba_v1 = v1_model.predict_proba(X_val_v1)[:, 1]
    v1_metrics = evaluate_and_save(y_val_v1, y_proba_v1, V1_THRESHOLD,
                                    'V1', ['Non-Crisis', 'Crisis'])

    # Save final V1 metrics
    joblib.dump(v1_metrics, os.path.join(V1_DIR, "v1_final_metrics.pkl"))

    # ==================================================================
    # V2 FINAL EVALUATION
    # ==================================================================
    print("\n\n" + "=" * 60)
    print(f"V2 FINAL RESULTS (threshold={V2_THRESHOLD})")
    print("=" * 60)

    v2_model = joblib.load(os.path.join(V1_DIR, "v2_xgb_model.pkl"))
    v2_features = joblib.load(os.path.join(V1_DIR, "v2_feature_columns.pkl"))

    # Load and filter window data
    frames = []
    for year in [2019, 2020, 2021, 2022]:
        df = pd.read_csv(os.path.join(BASE_DIR, f"processed_{year}_modeling_ready.csv"))
        df['year'] = year
        frames.append(df)
    df_win = pd.concat(frames, ignore_index=True)

    # Remove post-crisis windows
    is_crisis = df_win['is_crisis_user'] == 1
    is_pre = df_win['days_to_crisis'] > 0
    is_non = df_win['is_crisis_user'] == 0
    df_win = df_win[is_non | (is_crisis & is_pre)].copy()

    EXCLUDE = ['author', 'window_start_time', 'window_end_time',
               'label', 'is_crisis_user', 'days_to_crisis', 'year']
    feat_cols = [c for c in df_win.columns if c not in EXCLUDE]

    val_mask_v2 = df_win['author'].isin(val_users_set)
    X_val_v2 = df_win.loc[val_mask_v2, feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val_v2 = df_win.loc[val_mask_v2, 'label']

    print(f"Validation windows: {val_mask_v2.sum():,}")
    print(f"Positive windows in val: {(y_val_v2 == 1).sum():,} ({y_val_v2.mean()*100:.2f}%)")

    y_proba_v2 = v2_model.predict_proba(X_val_v2)[:, 1]
    v2_metrics = evaluate_and_save(y_val_v2, y_proba_v2, V2_THRESHOLD,
                                    'V2', ['Normal', 'Pre-Crisis'])

    # Save final V2 metrics
    joblib.dump(v2_metrics, os.path.join(V1_DIR, "v2_final_metrics.pkl"))

    # ==================================================================
    # SIDE-BY-SIDE COMPARISON
    # ==================================================================
    print("\n\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON: DEFAULT vs OPTIMIZED")
    print("=" * 60)

    print(f"\n{'':25} {'V1 (User-Level)':>30}  {'V2 (Temporal)':>30}")
    print(f"{'':25} {'Default':>14} {'Optimized':>14}  {'Default':>14} {'Optimized':>14}")
    print("-" * 100)
    print(f"{'Threshold':<25} {'0.50':>14} {V1_THRESHOLD:>14.2f}  {'0.50':>14} {V2_THRESHOLD:>14.2f}")
    print(f"{'Recall':<25} {'0.6273':>14} {v1_metrics['recall']:>14.4f}  {'0.0802':>14} {v2_metrics['recall']:>14.4f}")
    print(f"{'Precision':<25} {'0.8742':>14} {v1_metrics['precision']:>14.4f}  {'0.2234':>14} {v2_metrics['precision']:>14.4f}")
    print(f"{'F1 Score':<25} {'0.7304':>14} {v1_metrics['f1']:>14.4f}  {'0.1180':>14} {v2_metrics['f1']:>14.4f}")
    print(f"{'AUROC':<25} {'0.8788':>14} {v1_metrics['auroc']:>14.4f}  {'0.8842':>14} {v2_metrics['auroc']:>14.4f}")
    print(f"{'AUPRC':<25} {'0.8688':>14} {v1_metrics['auprc']:>14.4f}  {'0.1603':>14} {v2_metrics['auprc']:>14.4f}")
    print(f"{'Missed (FN)':<25} {'1,618':>14} {v1_metrics['FN']:>14,}  {'241':>14} {v2_metrics['FN']:>14,}")
    print(f"{'Caught (TP)':<25} {'2,723':>14} {v1_metrics['TP']:>14,}  {'21':>14} {v2_metrics['TP']:>14,}")

    # Save thresholds for future use
    thresholds = {
        'v1_threshold': V1_THRESHOLD,
        'v2_threshold': V2_THRESHOLD,
        'v1_metrics': v1_metrics,
        'v2_metrics': v2_metrics,
    }
    joblib.dump(thresholds, os.path.join(V1_DIR, "optimized_thresholds.pkl"))

    print(f"\n\nSaved: v1_final_metrics.pkl")
    print(f"Saved: v2_final_metrics.pkl")
    print(f"Saved: optimized_thresholds.pkl")
    print(f"Saved: final_evaluation.log")
    print(f"\n✅ FINAL EVALUATION COMPLETE.")


if __name__ == "__main__":
    main()
