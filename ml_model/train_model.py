




from preprocessing import preprocess_data
from feature_engineering import engineer_features



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv(r"C:\Users\tanishka\Downloads\DATASET_CC\construction_project_dataset.csv")

# Step 1 & 2
df = preprocess_data(df)

# Step 3
df = engineer_features(df)

# 🔥 Step 5: Define X and y
X = df.drop('total_gwp_kgCO2e', axis=1)
y = df['total_gwp_kgCO2e']

# 🔥 Step 6: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 🔥 MODEL TRAINING
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔥 PREDICTION
y_pred = model.predict(X_test)

# 🔥 EVALUATION
from sklearn.metrics import mean_squared_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 🔥 SAVE MODEL
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns, "model_columns.pkl")

print("Model saved successfully")
