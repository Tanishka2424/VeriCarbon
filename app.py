import streamlit as st

from ml_model.predict import predict_emission
from ml_model.predict_anomaly import check_validity

st.set_page_config(page_title="Carbon Credit System")

st.title("🌍 Carbon Credit Verification System")

st.write("Enter project details below:")

# 🔷 INPUTS
quantity = st.number_input("Quantity (kg)", min_value=0.0)
floor_area = st.number_input("Floor Area (m²)", min_value=0.0)
floors = st.number_input("Number of Floors", min_value=1)

building_type = st.selectbox("Building Type", [
    "Residential", "Commercial", "Industrial"
])

region = st.selectbox("Region", [
    "India", "USA", "Europe"
])

lifecycle_stage = st.selectbox("Lifecycle Stage", [
    "A1-Raw Material",
    "A2-Transport",
    "A3-Manufacturing"
])

# 🔷 BUTTON
if st.button("Check & Predict"):

    input_data = {
        'quantity_kg': quantity,
        'floor_area_m2': floor_area,
        'num_floors': floors,
        'ec_per_kg': 0.5,
        'building_type': building_type,
        'region': region,
        'lifecycle_stage': lifecycle_stage
    }

    # 🔷 STEP 1: VALIDATION
    validity = check_validity(input_data)

    if "Suspicious" in validity:
        st.error(f"❌ Data flagged as suspicious!\n{validity}")
    else:
        st.success("✅ Data is valid")

        # 🔷 STEP 2: EMISSION PREDICTION
        emission = predict_emission(input_data)

        st.subheader("🌱 Emission Result")
        st.write(f"Estimated CO₂ Emission: **{emission:.2f} kg**")

        # 🔷 STEP 3: CREDIT LOGIC (simple)
        credits = max(0, 10000 - emission) / 100

        st.subheader("💰 Carbon Credits")
        st.write(f"Credits Earned: **{credits:.2f} tokens**")