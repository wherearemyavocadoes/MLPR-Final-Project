# V2 Undersampling Experiment

## Why We Tested This

The V2 temporal model has extremely imbalanced data: only **2.59% of windows are positive** (pre-crisis). To handle this, we used **class weighting** (`scale_pos_weight=37.6`), which tells the model that missing a crisis is 37.6× worse than a false alarm.

Our mentor suggested also testing **undersampling** — reducing the number of normal windows to create a less imbalanced training set. A previous experiment with 50/50 undersampling failed badly (precision collapsed to 3.4%), so we tested a milder ratio: **85% negative / 15% positive**.

---

## What We Did

- Used the **same data, same user split, same model** as V2
- Only change: undersampled the training set from 33,090 windows to 5,720 windows
  - Kept all 858 positive windows
  - Reduced negative windows from 32,232 to 4,862 (removed 84.9%)
- Trained XGBoost with the same parameters
- Found optimal threshold and compared

---

## Results

### At each model's best-F1 threshold:

| Metric | Current V2 (class weighting, thresh=0.06) | Undersampled 85/15 (thresh=0.53) |
|--|--|--|
| **Recall** | **0.714** | 0.470 |
| Precision | 0.118 | 0.140 |
| F1 | 0.202 | 0.216 |
| AUROC | **0.884** | 0.874 |
| Missed (FN) | **75** | 139 |
| Caught (TP) | **187** | 123 |

### At matched recall (~70%):

| | Current V2 | Undersampled |
|--|--|--|
| Threshold | 0.06 | 0.30 |
| Recall | 0.714 | 0.710 |
| **Precision** | **0.118** | **0.118** |

At matched recall levels, **both approaches give identical precision**. Neither is fundamentally better — they operate at different points on the same precision-recall curve.

### What if undersampled model uses a very low threshold (~0.05)?

| | Current V2 @ 0.06 | Undersampled @ 0.05 |
|--|--|--|
| Recall | 0.714 | **0.950** |
| Precision | **0.118** | 0.087 |
| False alarms | 1,403 | ~2,500+ |

The undersampled model CAN push recall to 95%, but at the cost of much lower precision and roughly double the false alarms.

---

## Conclusion

**At matched recall, both methods perform identically.** The difference is only in how far you can push recall:

- **Current approach (class weighting):** Safer. Uses all training data. AUROC slightly higher (0.884 vs 0.874). Good balance at 71.4% recall.
- **Undersampled 85/15 + low threshold:** Can push recall to 95%, but with more false alarms and lower precision.

The choice depends on the use case:
- If **minimizing missed crises is the top priority** (every person matters), undersampling + low threshold catches more.
- If **keeping false alarms manageable matters**, class weighting is cleaner.

This decision is pending mentor feedback.

---

## Files

| File | Description |
|------|-------------|
| `undersampling_experiment.py` | Full experiment script |
| `undersampling_experiment.log` | All results and comparison tables |