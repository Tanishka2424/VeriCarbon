import pandas as pd
import joblib

from preprocessing import preprocess_data
from feature_engineering import engineer_features

from sklearn.ensemble import IsolationForest

# 🔷 Load data
df = pd.read_csv(r"C:\Users\tanishka\Downloads\DATASET_CC\construction_project_dataset.csv")

# 🔷 Preprocess
df = preprocess_data(df)

# 🔷 Feature Engineering
df = engineer_features(df)

# 🔷 Remove target column
X = df.drop('total_gwp_kgCO2e', axis=1)

# 🔷 Train Anomaly Model
model = IsolationForest(
    n_estimators=100,
    contamination=0.15,
    random_state=42
)

model.fit(X)

# 🔷 Save model
joblib.dump(model, "anomaly_model.pkl")
joblib.dump(X.columns, "anomaly_columns.pkl")

print("Anomaly model trained and saved!")

# 🔷 OPTIONAL TEST
preds = model.predict(X[:10])

print("Sample predictions:", preds)