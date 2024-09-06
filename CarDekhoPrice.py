import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model, scaler, encoder, and sample data
with open(r'C:\Users\geeth\OneDrive\Desktop\GUVI Projects\CarDekho\Code files\CarDekhoPrice_Model.pkl', 'rb') as file:
    model = pickle.load(file)

with open(r'C:\Users\geeth\OneDrive\Desktop\GUVI Projects\CarDekho\Code files\CarDekhoPrice_Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open(r'C:\Users\geeth\OneDrive\Desktop\GUVI Projects\CarDekho\Code files\CarDekhoPrice_Encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Example data frame to extract unique values for categorical features
# Replace with the data used during model training for actual implementation
reference_data = pd.read_excel(r'C:\Users\geeth\OneDrive\Desktop\GUVI Projects\CarDekho\Code files\FeaturesEngineered.xlsx')  # Load reference data to get unique options

important_numerical_cols = ['Width', 'MaxPower', 'ManufactureYear', 'KilometersDriven', 'Length',
                            'WheelBase', 'KerbWeight', 'Torque', 'Engine', 'Mileage',
                             'PreviousOwners', 'Seats', 'Doors', 'Car_Age', 'TopSpeed']
important_categorical_cols = ['City', 'FuelType', 'BodyType', 'manufacturer', 'CarModel', 'Color',
                              'EngineType', 'TransmissionType', 'DriveType', 'FuelSupplySystem', 'RearBrakeType',
                              'TyreType', 'SteeringType','Locking', 'GearBox']

# Extract min and max values for numerical inputs from the reference data
numerical_ranges = {
    'Width': (reference_data['Width'].min(), reference_data['Width'].max()),
    'MaxPower': (reference_data['MaxPower'].min(), reference_data['MaxPower'].max()),
    'ManufactureYear': (reference_data['ManufactureYear'].min(), reference_data['ManufactureYear'].max()),
    'KilometersDriven': (reference_data['KilometersDriven'].min(), reference_data['KilometersDriven'].max()),
    'Length': (reference_data['Length'].min(), reference_data['Length'].max()),
    'WheelBase': (reference_data['WheelBase'].min(), reference_data['WheelBase'].max()),
    'KerbWeight': (reference_data['KerbWeight'].min(), reference_data['KerbWeight'].max()),
    'Torque': (reference_data['Torque'].min(), reference_data['Torque'].max()),
    'Engine': (reference_data['Engine'].min(), reference_data['Engine'].max()),
    'Mileage': (reference_data['Mileage'].min(), reference_data['Mileage'].max()),
    'PreviousOwners': (int(reference_data['PreviousOwners'].min()), int(reference_data['PreviousOwners'].max())),
    'Seats': (int(reference_data['Seats'].min()), int(reference_data['Seats'].max())),
    'Doors': (int(reference_data['Doors'].min()), int(reference_data['Doors'].max())),
    'Car_Age': (int(reference_data['Car_Age'].min()), int(reference_data['Car_Age'].max())),
    'TopSpeed': (reference_data['TopSpeed'].min(), reference_data['TopSpeed'].max()),
}

# Extract unique options from the data
cities = reference_data['City'].unique().tolist()
fuel_types = reference_data['FuelType'].unique().tolist()
body_types = reference_data['BodyType'].unique().tolist()
manufacturers = reference_data['manufacturer'].unique().tolist()
car_models = reference_data['CarModel'].unique().tolist()
colors = reference_data['Color'].unique().tolist()
engine_types = reference_data['EngineType'].unique().tolist()
transmission_types = reference_data['TransmissionType'].unique().tolist()
drive_types = reference_data['DriveType'].unique().tolist()
fuel_supply_systems = reference_data['FuelSupplySystem'].unique().tolist()
rear_brake_types = reference_data['RearBrakeType'].unique().tolist()
tyre_types = reference_data['TyreType'].unique().tolist()
steering_types = reference_data['SteeringType'].unique().tolist()
head_lights = reference_data['HeadLights'].unique().tolist()
lockings = reference_data['Locking'].unique().tolist()
gear_boxes = reference_data['GearBox'].unique().tolist()

# Streamlit UI
st.title("Car Price Prediction App")
st.write("Enter the car features below and get an estimated price.")

# Apply custom background
background_css = '''
    <style>
    .stApp {
        background-image:url("https://img.freepik.com/premium-photo/nighttime-view-sports-car-city-highway-seen-from-with-urban-landscape-city-lights-background-dynamic-thrilling-scene-capturing-speed-energy-night-drive_256259-7609.jpg?w=1060");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
'''
st.markdown(background_css, unsafe_allow_html=True)

# Define feature input function
def get_user_input():
    st.sidebar.header("Enter Numerical Features")

    #Numerical features input with dynamic ranges
    width = st.sidebar.number_input("Width", min_value=numerical_ranges['Width'][0], max_value=numerical_ranges['Width'][1], value=numerical_ranges['Width'][0])
    max_power = st.sidebar.number_input("Max Power (hp)", min_value=numerical_ranges['MaxPower'][0], max_value=numerical_ranges['MaxPower'][1], value=numerical_ranges['MaxPower'][0])
    manufacture_year = st.sidebar.slider("Manufacture Year", min_value=int(numerical_ranges['ManufactureYear'][0]), max_value=int(numerical_ranges['ManufactureYear'][1]), value=int(numerical_ranges['ManufactureYear'][0]))
    kilometers_driven = st.sidebar.number_input("Kilometers Driven", min_value=numerical_ranges['KilometersDriven'][0], max_value=numerical_ranges['KilometersDriven'][1], value=numerical_ranges['KilometersDriven'][0])
    length = st.sidebar.number_input("Length (mm)", min_value=numerical_ranges['Length'][0], max_value=numerical_ranges['Length'][1], value=numerical_ranges['Length'][0])
    wheel_base = st.sidebar.number_input("Wheel Base (mm)", min_value=numerical_ranges['WheelBase'][0], max_value=numerical_ranges['WheelBase'][1], value=numerical_ranges['WheelBase'][0])
    kerb_weight = st.sidebar.number_input("Kerb Weight (kg)", min_value=numerical_ranges['KerbWeight'][0], max_value=numerical_ranges['KerbWeight'][1], value=numerical_ranges['KerbWeight'][0])
    torque = st.sidebar.number_input("Torque (Nm)", min_value=numerical_ranges['Torque'][0], max_value=numerical_ranges['Torque'][1], value=numerical_ranges['Torque'][0])
    engine = st.sidebar.number_input("Engine Capacity (cc)", min_value=numerical_ranges['Engine'][0], max_value=numerical_ranges['Engine'][1], value=numerical_ranges['Engine'][0])
    mileage = st.sidebar.number_input("Mileage (kmpl)", min_value=numerical_ranges['Mileage'][0], max_value=numerical_ranges['Mileage'][1], value=numerical_ranges['Mileage'][0])
    previous_owners = st.sidebar.slider("Previous Owners", min_value=numerical_ranges['PreviousOwners'][0], max_value=numerical_ranges['PreviousOwners'][1], value=numerical_ranges['PreviousOwners'][0])
    seats = st.sidebar.selectbox("Number of Seats", options=range(int(numerical_ranges['Seats'][0]), int(numerical_ranges['Seats'][1]) + 1), index=0 )
    doors = st.sidebar.slider("Number of Doors", min_value=numerical_ranges['Doors'][0], max_value=numerical_ranges['Doors'][1], value=numerical_ranges['Doors'][0])
    car_age = st.sidebar.slider("Car Age (years)", min_value=numerical_ranges['Car_Age'][0], max_value=numerical_ranges['Car_Age'][1], value=numerical_ranges['Car_Age'][0])
    top_speed = st.sidebar.number_input("Top Speed (km/h)", min_value=numerical_ranges['TopSpeed'][0], max_value=numerical_ranges['TopSpeed'][1], value=numerical_ranges['TopSpeed'][0])

    # Categorical features input
    city = st.selectbox("City", cities)
    fuel_type = st.selectbox("Fuel Type", fuel_types)
    body_type = st.selectbox("Body Type", body_types)
    manufacturer = st.selectbox("Manufacturer", manufacturers)
    car_model = st.selectbox("Car Model", car_models)  # Selecting from available models
    variant_name = st.text_input("Variant Name", "VariantY")  # Example placeholder
    color = st.selectbox("Color", colors)
    engine_type = st.selectbox("Engine Type", engine_types)
    transmission_type = st.selectbox("Transmission Type", transmission_types)
    drive_type = st.selectbox("Drive Type", drive_types)
    fuel_supply_system = st.selectbox("Fuel Supply System", fuel_supply_systems)
    rear_brake_type = st.selectbox("Rear Brake Type", rear_brake_types)
    tyre_type = st.selectbox("Tyre Type", tyre_types)
    steering_type = st.selectbox("Steering Type", steering_types)
    locking = st.selectbox("Locking", lockings)
    gear_box = st.selectbox("Gear Box", gear_boxes)

    # Create a dictionary of inputs
    features = {
        'Width': width,
        'MaxPower': max_power,
        'ManufactureYear': manufacture_year,
        'KilometersDriven': kilometers_driven,
        'Length': length,
        'WheelBase': wheel_base,
        'KerbWeight': kerb_weight,
        'Torque': torque,
        'Engine': engine,
        'Mileage': mileage,
        'PreviousOwners': previous_owners,
        'Seats': seats,
        'Doors': doors,
        'Car_Age': car_age,
        'TopSpeed': top_speed,
        'City': city,
        'FuelType': fuel_type,
        'BodyType': body_type,
        'manufacturer': manufacturer,
        'CarModel': car_model,
        'VariantName': variant_name,
        'Color': color,
        'EngineType': engine_type,
        'TransmissionType': transmission_type,
        'DriveType': drive_type,
        'FuelSupplySystem': fuel_supply_system,
        'RearBrakeType': rear_brake_type,
        'TyreType': tyre_type,
        'SteeringType': steering_type,
        'Locking': locking,
        'GearBox': gear_box
    }

    return pd.DataFrame(features, index=[0])

# Predict function
def predict_price(features):
    # Get available columns from user input
    available_columns = features.columns.tolist()

    # Handle missing numerical columns by filling with mean or a suitable value
    for col in important_numerical_cols:
        if col not in available_columns:
            features[col] = 0  # Replace with mean or a default value if necessary

    # Handle missing categorical columns by setting to 'Unknown'
    for col in important_categorical_cols:
        if col not in available_columns:
            features[col] = 'Unknown'

    # Separate numerical and categorical features
    numerical_features = features[important_numerical_cols]
    categorical_features = features[important_categorical_cols]

    # Scale numerical features
    numerical_features_scaled = scaler.transform(numerical_features)

    # Encode categorical features
    categorical_features_encoded = encoder.transform(categorical_features).toarray()

    # Combine scaled and encoded features
    features_processed = np.hstack([numerical_features_scaled, categorical_features_encoded])

    # Predict the price
    prediction = model.predict(features_processed)

    return prediction[0]

# Get user input
user_input = get_user_input()

# Prediction button
if st.button("Predict Car Price"):
    try:
        prediction = predict_price(user_input)
        st.success(f"The estimated price of the car is: {prediction:.2f}Lakhs")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
