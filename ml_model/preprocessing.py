import pandas as pd


def preprocess_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.bfill().ffill()

    # Select important columns
    df = df[[
        'material_name',
        'quantity_kg',
        'building_type',
        'region',
        'lifecycle_stage',
        'num_floors',
        'floor_area_m2',
        'ec_per_kg',
        'total_gwp_kgCO2e'
    ]]

    # Encoding (categorical -> numeric)
    df = pd.get_dummies(
        df,
        columns=[
            'material_name',
            'building_type',
            'region',
            'lifecycle_stage'
        ],
        drop_first=True
    )

    return df