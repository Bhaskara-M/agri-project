import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import joblib

# Step 1: Generate synthetic dataset (replace with real CSV if available)
np.random.seed(42)
n_samples = 500

data = {
    "ph": np.random.uniform(3, 10, n_samples),
    "n": np.random.uniform(0, 300, n_samples),
    "p": np.random.uniform(0, 200, n_samples),
    "k": np.random.uniform(0, 300, n_samples),
    "ec": np.random.uniform(0, 4, n_samples),
    "organic_matter": np.random.uniform(0, 10, n_samples),
    "moisture": np.random.uniform(0, 100, n_samples),
    "fe": np.random.uniform(0, 10, n_samples),
    "zn": np.random.uniform(0, 10, n_samples),
    "mn": np.random.uniform(0, 10, n_samples),
    "cu": np.random.uniform(0, 10, n_samples),
    "b": np.random.uniform(0, 10, n_samples),
}

df = pd.DataFrame(data)

# Step 2: Define target (rule-based label for demo)
df["label"] = (
    (df["ph"].between(6, 7.5)) &
    (df["n"] > 50) &
    (df["p"] > 30) &
    (df["k"] > 50) &
    (df["organic_matter"] > 1)
).astype(int)

# Step 3: Split data
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# Step 6: Predictions
y_pred = mlp.predict(X_test_scaled)
y_prob = mlp.predict_proba(X_test_scaled)[:, 1]

# Step 7: Metrics
print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("🔹 Precision:", precision_score(y_test, y_pred))
print("🔹 Recall:", recall_score(y_test, y_pred))
print("🔹 F1 Score:", f1_score(y_test, y_pred))
print("🔹 ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save model and scaler
joblib.dump(mlp, "soil_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("\n✅ MLP model and scaler saved successfully!")