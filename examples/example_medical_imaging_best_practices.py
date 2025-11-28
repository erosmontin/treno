"""
Example: Medical Imaging Best Practices - Group-Aware Splitting

This example demonstrates critical utilities for medical imaging ML:
- Patient-aware data splitting (prevents leakage)
- Comprehensive binary metrics (clinical statistics)
- Proper handling of data augmentation
"""

import numpy as np
import pandas as pd
from treno import (
    stratified_group_split,
    extract_patient_groups,
    compute_binary_metrics,
    EMUNet
)

# ============================================================================
# Simulate Medical Imaging Dataset with Data Augmentation
# ============================================================================

# Create patient IDs (20 patients)
patient_ids = [f'patient_{i:03d}' for i in range(20)]

# Add augmented versions (3 augmentations per patient)
all_ids = []
for pid in patient_ids:
    all_ids.append(pid)  # Original
    all_ids.extend([f'{pid}-aug{j}' for j in range(1, 4)])  # Augmentations

print(f"Total samples: {len(all_ids)} (20 patients × 4 versions)")

# Simulate features and labels
X = pd.DataFrame(
    np.random.randn(len(all_ids), 50),  # 50 radiomics features
    index=all_ids
)

# Labels (some correlation with patient ID for realism)
y = pd.Series(
    [int(int(pid.split('_')[1]) % 2) for pid in all_ids],
    index=all_ids
)

print(f"\nClass distribution: {y.value_counts().to_dict()}")

# ============================================================================
# WRONG WAY (Data Leakage!)
# ============================================================================
print("\n" + "="*70)
print("❌ WRONG: Random split (causes data leakage)")
print("="*70)

from sklearn.model_selection import train_test_split

X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Check if same patient appears in both sets
train_patients = set([idx.split('-aug')[0] for idx in X_train_wrong.index])
test_patients = set([idx.split('-aug')[0] for idx in X_test_wrong.index])
overlap = train_patients.intersection(test_patients)

print(f"Train patients: {len(train_patients)}")
print(f"Test patients: {len(test_patients)}")
print(f"⚠️  OVERLAP: {len(overlap)} patients appear in BOTH train and test!")
print(f"   This causes DATA LEAKAGE - model sees test patients during training!")

# ============================================================================
# CORRECT WAY (Patient-Aware Split)
# ============================================================================
print("\n" + "="*70)
print("✅ CORRECT: Group-aware split (no leakage)")
print("="*70)

X_train, X_test, y_train, y_test, groups_train, groups_test = stratified_group_split(
    X, y,
    test_size=0.25,
    random_state=42,
    augmentation_suffix='-aug'
)

# Check patient overlap
train_patients = set([idx.split('-aug')[0] for idx in X_train.index])
test_patients = set([idx.split('-aug')[0] for idx in X_test.index])
overlap = train_patients.intersection(test_patients)

print(f"Train patients: {len(train_patients)}")
print(f"Test patients: {len(test_patients)}")
print(f"✅ OVERLAP: {len(overlap)} patients (perfect!)")
print(f"   Train/test are completely independent!")

print(f"\nTrain samples: {len(X_train)} (includes augmentations)")
print(f"Test samples: {len(X_test)} (includes augmentations)")
print(f"Class balance maintained: train={y_train.value_counts().to_dict()}, "
      f"test={y_test.value_counts().to_dict()}")

# ============================================================================
# Train a Model and Evaluate with Comprehensive Metrics
# ============================================================================
print("\n" + "="*70)
print("Model Training & Evaluation")
print("="*70)

# Simple sklearn model for demo
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predictions
y_pred = clf.predict(X_test_scaled)

# ============================================================================
# Comprehensive Binary Metrics (Clinical Statistics)
# ============================================================================
print("\n" + "="*70)
print("📊 Comprehensive Clinical Metrics")
print("="*70)

metrics = compute_binary_metrics(y_test, y_pred)

print("\n🎯 Classification Performance:")
print(f"  Accuracy:    {metrics['accuracy']:.3f}")
print(f"  Sensitivity: {metrics['sensitivity']:.3f} (recall, true positive rate)")
print(f"  Specificity: {metrics['specificity']:.3f} (true negative rate)")
print(f"  Precision:   {metrics['precision']:.3f}")
print(f"  F1 Score:    {metrics['f1']:.3f}")

print("\n🔬 Clinical Measures:")
print(f"  Odds Ratio:      {metrics['odds_ratio']:.3f}")
print(f"  Relative Risk:   {metrics['relative_risk']:.3f}")
print(f"  MCC:             {metrics['mcc']:.3f} (Matthews Correlation)")

print("\n📈 ROC Analysis:")
print(f"  AUC:             {metrics['auc']:.3f}")
print(f"  Optimal Thresh:  {metrics['auc_threshold']:.3f}")

print("\n📋 Confusion Matrix Breakdown:")
print(f"  True Positives:  {metrics['true_positives']:.3f} (proportion)")
print(f"  True Negatives:  {metrics['true_negatives']:.3f}")
print(f"  False Positives: {metrics['false_positives']:.3f}")
print(f"  False Negatives: {metrics['false_negatives']:.3f}")

# ============================================================================
# Key Takeaways
# ============================================================================
print("\n" + "="*70)
print("🎓 KEY TAKEAWAYS")
print("="*70)
print("""
1. ✅ ALWAYS use stratified_group_split() for medical imaging
   - Prevents patient data leakage
   - Handles augmented samples correctly
   - Maintains class balance

2. ✅ Use compute_binary_metrics() for clinical reporting
   - Includes sensitivity/specificity (clinical standard)
   - Provides odds ratio and relative risk
   - Gives MCC (better than accuracy for imbalanced data)

3. ✅ Extract patient groups with extract_patient_groups()
   - Automatically handles '-aug' suffixes
   - Works with any grouping scheme
   - Compatible with sklearn's GroupKFold

4. ⚠️  NEVER use random train_test_split() directly
   - Causes data leakage in medical imaging
   - Inflates performance metrics
   - Results won't generalize to new patients!
""")

# ============================================================================
# Bonus: Using with Cross-Validation
# ============================================================================
print("\n" + "="*70)
print("Bonus: Cross-Validation with Patient Groups")
print("="*70)

from sklearn.model_selection import cross_val_score

# Extract groups for all data
groups = extract_patient_groups(X.index, augmentation_suffix='-aug')

# Use GroupKFold for cross-validation
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    clf, 
    scaler.fit_transform(X), 
    y, 
    cv=sgkf, 
    groups=groups,
    scoring='accuracy'
)

print(f"\n5-Fold Group Cross-Validation:")
print(f"  Scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean:   {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  ✅ Each fold has completely different patients!")

print("\n✨ Your model is now properly validated for medical imaging!")
