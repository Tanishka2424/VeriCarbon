import joblib
import pandas as pd

# 🔷 Load model artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


def predict_emission(input_data: dict):
    """
    Predict carbon emission from input data
    """

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # 🔷 Feature Engineering
    df['energy_consumption'] = df['quantity_kg'] * 0.5
    df['machine_hours'] = df['num_floors'] * 10
    df['transport_distance_km'] = 50

    # 🔷 Encoding
    df = pd.get_dummies(df)

    # 🔷 Align columns with training data
    df = df.reindex(columns=model_columns, fill_value=0)

    # 🔷 Convert all to numeric
    df = df.astype(float)

    # 🔷 Scaling
    df_scaled = scaler.transform(df)

    # 🔷 Prediction
    prediction = model.predict(df_scaled)

    return prediction[0]


# 🔷 TEST BLOCK
if __name__ == "__main__":
    sample_input = {
        'quantity_kg': 5000,
        'floor_area_m2': 2000,
        'num_floors': 5,
        'ec_per_kg': 0.5,
        'building_type': 'Residential',
        'region': 'India',
        'lifecycle_stage': 'A3-Manufacturing'
    }

    result = predict_emission(sample_input)
    print("Predicted Emission:", result)