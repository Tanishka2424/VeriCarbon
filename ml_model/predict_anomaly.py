import joblib
import pandas as pd

# Load model + columns
model = joblib.load("anomaly_model.pkl")
model_columns = joblib.load("anomaly_columns.pkl")


def check_validity(input_data: dict):
     # 🔥 RULE-BASED CHECKS (ADD HERE)
    if input_data['quantity_kg'] <= 0:
        return "Suspicious ❌"

    if input_data['floor_area_m2'] <= 0:
        return "Suspicious ❌"

    if input_data['num_floors'] > 200:
        return "Suspicious ❌"

    if input_data['quantity_kg'] > 1e7:
        return "Suspicious ❌"

    if input_data['ec_per_kg'] > 10:
        return "Suspicious ❌"
    



    #normal flow
    df = pd.DataFrame([input_data])

    # Feature Engineering
    df['energy_consumption'] = df['quantity_kg'] * 0.5
    df['machine_hours'] = df['num_floors'] * 10
    df['transport_distance_km'] = 50

    # Encoding
    df = pd.get_dummies(df)

    # 🔥 IMPORTANT FIX (same as model 1)
    df = df.reindex(columns=model_columns, fill_value=0)

    # Convert to numeric
    df = df.astype(float)

    # Prediction
    result = model.predict(df)

    if result[0] == 1:
        return "Valid ✅"
    else:
        return "Suspicious ❌"


# TEST
if __name__ == "__main__":

    # ✅ Normal case
    sample_valid = {
        'quantity_kg': 5000,
        'floor_area_m2': 2000,
        'num_floors': 5,
        'ec_per_kg': 0.5,
        'building_type': 'Residential',
        'region': 'India',
        'lifecycle_stage': 'A3-Manufacturing'
    }

    # ❌ Suspicious case
    sample_suspicious = {
        'quantity_kg': 100000000,
        'floor_area_m2': 50,
        'num_floors': 1,
        'ec_per_kg': 0.5,
        'building_type': 'Residential',
        'region': 'India',
        'lifecycle_stage': 'A3-Manufacturing'
    }

    print("Valid Test:", check_validity(sample_valid))
    print("Suspicious Test:", check_validity(sample_suspicious))

    