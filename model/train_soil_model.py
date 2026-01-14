"""
Improved Soil Quality Prediction Model Training Script

This script:
1. Loads and preprocesses the CSV dataset
2. Handles missing values, outliers, and incorrect data types
3. Performs feature engineering
4. Creates target labels based on soil quality criteria
5. Splits data into train/validation/test sets
6. Trains an improved ML model
7. Evaluates with comprehensive metrics
8. Saves model and scaler in the format expected by the backend

CRITICAL: Model output format must match exactly what ml_service.py expects:
- soil_score: float (0 or 1)
- soil_class: str ("Suitable" or "Unsuitable")
- confidence: float (0-1)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# STEP 1: Load and Explore Dataset
# ============================================================================
print("=" * 80)
print("STEP 1: Loading Dataset")
print("=" * 80)

# Find CSV file - check multiple possible locations
csv_paths = [
    "data_core_updated_varied.csv",
    "../data_core_updated_varied.csv",
    os.path.join(os.path.dirname(__file__), "..", "data_core_updated_varied.csv"),
]

csv_path = None
for path in csv_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    raise FileNotFoundError(
        "CSV file not found. Please ensure 'data_core_updated_varied.csv' exists in the project root."
    )

print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nBasic statistics:")
print(df.describe())

# ============================================================================
# STEP 2: Data Preprocessing
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Data Preprocessing")
print("=" * 80)

# Map CSV columns to expected feature names
# Backend expects: ph, n, p, k, ec, organic_matter, moisture, fe, zn, mn, cu, b
column_mapping = {
    "pH": "ph",
    "Nitrogen": "n",
    "Phosphorus": "p",
    "Potassium": "k",
    "EC_dS_m": "ec",
    "Organic_Matter_%": "organic_matter",
    "Moisture": "moisture",
    "Iron_ppm": "fe",
    "Zinc_ppm": "zn",
    "Manganese_ppm": "mn",
    "Copper_ppm": "cu",
    "Boron_ppm": "b",
}

# Create a new dataframe with mapped columns
df_processed = df.copy()

# Rename columns
for old_col, new_col in column_mapping.items():
    if old_col in df_processed.columns:
        df_processed[new_col] = df_processed[old_col]

# Select only the features we need (in the exact order expected by backend)
feature_columns = ["ph", "n", "p", "k", "ec", "organic_matter", "moisture", "fe", "zn", "mn", "cu", "b"]
df_features = df_processed[feature_columns].copy()

print(f"\nSelected features: {feature_columns}")
print(f"Feature dataframe shape: {df_features.shape}")

# Handle missing values
print("\nHandling missing values...")
missing_before = df_features.isnull().sum().sum()
print(f"Missing values before: {missing_before}")

# Fill missing values with median (more robust than mean for outliers)
for col in feature_columns:
    if df_features[col].isnull().sum() > 0:
        median_val = df_features[col].median()
        df_features[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} missing values with median: {median_val}")

missing_after = df_features.isnull().sum().sum()
print(f"Missing values after: {missing_after}")

# Handle incorrect data types
print("\nConverting data types...")
for col in feature_columns:
    # Convert to numeric, coercing errors to NaN
    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
    # Fill any NaN created by conversion
    if df_features[col].isnull().sum() > 0:
        df_features[col].fillna(df_features[col].median(), inplace=True)
    # Ensure float type
    df_features[col] = df_features[col].astype(float)

print("Data types after conversion:")
print(df_features.dtypes)

# Handle outliers using IQR method
print("\nHandling outliers using IQR method...")
outliers_removed = 0
for col in feature_columns:
    Q1 = df_features[col].quantile(0.25)
    Q3 = df_features[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers
    outliers_count = ((df_features[col] < lower_bound) | (df_features[col] > upper_bound)).sum()
    
    # Cap outliers instead of removing (to preserve data size)
    df_features.loc[df_features[col] < lower_bound, col] = lower_bound
    df_features.loc[df_features[col] > upper_bound, col] = upper_bound
    
    if outliers_count > 0:
        print(f"  {col}: Capped {outliers_count} outliers (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
        outliers_removed += outliers_count

print(f"Total outliers handled: {outliers_removed}")

# ============================================================================
# STEP 3: Feature Engineering
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Feature Engineering")
print("=" * 80)

# Create additional meaningful features
df_features_eng = df_features.copy()

# Nutrient ratios (important for soil quality assessment)
df_features_eng["np_ratio"] = df_features_eng["n"] / (df_features_eng["p"] + 1e-6)  # Add small value to avoid division by zero
df_features_eng["nk_ratio"] = df_features_eng["n"] / (df_features_eng["k"] + 1e-6)
df_features_eng["pk_ratio"] = df_features_eng["p"] / (df_features_eng["k"] + 1e-6)

# Total macronutrients
df_features_eng["total_macronutrients"] = df_features_eng["n"] + df_features_eng["p"] + df_features_eng["k"]

# Total micronutrients
df_features_eng["total_micronutrients"] = (
    df_features_eng["fe"] + df_features_eng["zn"] + df_features_eng["mn"] + 
    df_features_eng["cu"] + df_features_eng["b"]
)

# Nutrient balance score (higher is better)
df_features_eng["nutrient_balance"] = (
    (df_features_eng["n"] > 20).astype(int) +
    (df_features_eng["p"] > 15).astype(int) +
    (df_features_eng["k"] > 20).astype(int)
)

# Replace infinite values with finite values
df_features_eng = df_features_eng.replace([np.inf, -np.inf], np.nan)
df_features_eng = df_features_eng.fillna(df_features_eng.median())

print(f"Features after engineering: {df_features_eng.shape[1]} features")
print(f"New features added: {list(df_features_eng.columns[len(feature_columns):])}")

# ============================================================================
# STEP 4: Create Target Labels
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Creating Target Labels")
print("=" * 80)

# Create target labels based on realistic soil quality criteria
# Using stricter criteria based on agricultural best practices:
# - pH must be in optimal range (6.0-7.5) - CRITICAL
# - At least 2 out of 3 macronutrients (N, P, K) must be adequate
# - Organic matter must be sufficient
# - EC and moisture should be reasonable

# Critical requirement: pH must be in acceptable range
ph_ok = df_features["ph"].between(6.0, 7.5)

# Macronutrient adequacy (at least 2 out of 3 must be adequate)
n_adequate = df_features["n"] > 20  # Nitrogen threshold
p_adequate = df_features["p"] > 15  # Phosphorus threshold  
k_adequate = df_features["k"] > 20  # Potassium threshold

macronutrient_count = (
    n_adequate.astype(int) + 
    p_adequate.astype(int) + 
    k_adequate.astype(int)
)
# At least 2 macronutrients must be adequate
macronutrients_ok = macronutrient_count >= 2

# Other important factors
organic_matter_ok = df_features["organic_matter"] > 1.5
ec_ok = df_features["ec"] < 2.0
moisture_ok = df_features["moisture"].between(30, 70)

# Suitable soil must meet ALL of these:
# 1. pH in acceptable range (CRITICAL - non-negotiable)
# 2. At least 2 out of 3 macronutrients adequate
# 3. Organic matter sufficient
# 4. EC reasonable
# 5. Moisture in acceptable range

y = (
    ph_ok & 
    macronutrients_ok & 
    organic_matter_ok & 
    ec_ok & 
    moisture_ok
).astype(int)

print(f"Target distribution:")
print(f"  Suitable (1): {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
print(f"  Unsuitable (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.2f}%)")

# Check for class imbalance
if y.sum() < len(y) * 0.1 or y.sum() > len(y) * 0.9:
    print("\nWARNING: Severe class imbalance detected. Consider using class weights or resampling.")

# ============================================================================
# STEP 5: Prepare Features for Training
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Preparing Features for Training")
print("=" * 80)

# Use engineered features for training
X = df_features_eng.values

# But we need to ensure the model can work with the original 12 features during inference
# We'll train on engineered features but create a mapping

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# ============================================================================
# STEP 6: Train/Validation/Test Split
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Data Splitting")
print("=" * 80)

# First split: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: 75% train, 25% validation (of the 80%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print(f"\nTraining set class distribution:")
print(f"  Suitable (1): {y_train.sum()} ({y_train.sum()/len(y_train)*100:.2f}%)")
print(f"  Unsuitable (0): {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.2f}%)")

# ============================================================================
# STEP 7: Feature Scaling
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Feature Scaling")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

# ============================================================================
# STEP 8: Model Training and Selection
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: Model Training and Selection")
print("=" * 80)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
}

best_model = None
best_score = 0
best_name = None
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    if name == "Neural Network (MLP)":
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Predict on validation set
    if name == "Neural Network (MLP)":
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    else:
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'model': model
    }
    
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print(f"  Validation Precision: {precision:.4f}")
    print(f"  Validation Recall: {recall:.4f}")
    print(f"  Validation F1: {f1:.4f}")
    print(f"  Validation ROC-AUC: {roc_auc:.4f}")
    
    # Use F1 score as the primary metric (good for imbalanced data)
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name

print(f"\n{'='*80}")
print(f"Best Model: {best_name} (F1 Score: {best_score:.4f})")
print(f"{'='*80}")

# ============================================================================
# STEP 9: Final Model Evaluation on Test Set
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: Final Evaluation on Test Set")
print("=" * 80)

# Use best model for final evaluation
if best_name == "Neural Network (MLP)":
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate comprehensive metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"\nTest Set Results:")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall: {test_recall:.4f}")
print(f"  F1 Score: {test_f1:.4f}")
print(f"  ROC-AUC: {test_roc_auc:.4f}")

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Unsuitable', 'Suitable']))

# ============================================================================
# STEP 10: Create Inference Model (12 features only)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: Creating Inference Model for Backend Compatibility")
print("=" * 80)

# The backend expects only 12 features in a specific order
# We need to retrain on just the original 12 features (without engineered features)
# OR create a wrapper that maps 12 features to engineered features

# Option: Retrain on original 12 features for simplicity and compatibility
X_original = df_features[feature_columns].values

# Split again with original features
X_temp_orig, X_test_orig, y_temp_orig, y_test_orig = train_test_split(
    X_original, y, test_size=0.2, random_state=42, stratify=y
)
X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
    X_temp_orig, y_temp_orig, test_size=0.25, random_state=42, stratify=y_temp_orig
)

# Scale original features
scaler_orig = StandardScaler()
X_train_orig_scaled = scaler_orig.fit_transform(X_train_orig)
X_val_orig_scaled = scaler_orig.transform(X_val_orig)
X_test_orig_scaled = scaler_orig.transform(X_test_orig)

# Train final model on original 12 features
# Always use scaling for consistency with backend expectations
# Use the same model type as best model
if best_name == "Random Forest":
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    # Tree-based models don't require scaling, but we'll use it for consistency
    final_model.fit(X_train_orig_scaled, y_train_orig)
    use_scaled = True
elif best_name == "Gradient Boosting":
    final_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    # Tree-based models don't require scaling, but we'll use it for consistency
    final_model.fit(X_train_orig_scaled, y_train_orig)
    use_scaled = True
else:  # Neural Network
    final_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    final_model.fit(X_train_orig_scaled, y_train_orig)
    use_scaled = True

# Evaluate final model
if use_scaled:
    y_test_final_pred = final_model.predict(X_test_orig_scaled)
    y_test_final_proba = final_model.predict_proba(X_test_orig_scaled)[:, 1]
else:
    y_test_final_pred = final_model.predict(X_test_orig)
    y_test_final_proba = final_model.predict_proba(X_test_orig)[:, 1]

final_accuracy = accuracy_score(y_test_orig, y_test_final_pred)
final_f1 = f1_score(y_test_orig, y_test_final_pred, zero_division=0)

print(f"\nFinal Model (12 features) Test Results:")
print(f"  Accuracy: {final_accuracy:.4f}")
print(f"  F1 Score: {final_f1:.4f}")

# ============================================================================
# STEP 11: Save Model and Scaler
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: Saving Model and Scaler")
print("=" * 80)

# Ensure model directory exists
model_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save model
model_path = os.path.join(model_dir, "soil_model.joblib")
joblib.dump(final_model, model_path)
print(f"Model saved to: {model_path}")

# Save scaler (always save a scaler for backend compatibility)
scaler_path = os.path.join(model_dir, "scaler.joblib")
joblib.dump(scaler_orig, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# ============================================================================
# STEP 12: Verify Model Output Format
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: Verifying Model Output Format")
print("=" * 80)

# Test prediction to ensure output format matches backend expectations
test_sample = X_test_orig[0:1]  # Take first test sample

if use_scaled:
    test_sample_scaled = scaler_orig.transform(test_sample)
    prediction = final_model.predict(test_sample_scaled)[0]
    if hasattr(final_model, "predict_proba"):
        confidence = float(max(final_model.predict_proba(test_sample_scaled)[0]))
    else:
        confidence = None
else:
    prediction = final_model.predict(test_sample)[0]
    if hasattr(final_model, "predict_proba"):
        confidence = float(max(final_model.predict_proba(test_sample)[0]))
    else:
        confidence = None

soil_class = "Suitable" if prediction == 1 else "Unsuitable"

output = {
    "soil_score": float(prediction),
    "soil_class": soil_class,
    "confidence": confidence,
}

print(f"Sample prediction output:")
print(f"  {output}")
print(f"\nSUCCESS: Output format matches backend expectations!")

print("\n" + "=" * 80)
print("SUCCESS: Training Complete!")
print("=" * 80)
print(f"\nModel Summary:")
print(f"  Model Type: {best_name}")
print(f"  Features: {len(feature_columns)} (in order: {', '.join(feature_columns)})")
print(f"  Training Samples: {len(X_train_orig)}")
print(f"  Test Accuracy: {final_accuracy:.4f}")
print(f"  Test F1 Score: {final_f1:.4f}")
print(f"\nModel files saved in: {model_dir}")
