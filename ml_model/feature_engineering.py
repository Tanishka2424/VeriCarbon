def engineer_features(df):
    
    
    
  


    # Dummy features
    df['energy_consumption'] = df['quantity_kg'] * 0.5
    df['machine_hours'] = df['num_floors'] * 10
    df['transport_distance_km'] = 50

    return df

