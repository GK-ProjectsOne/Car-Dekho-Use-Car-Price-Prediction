import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix

# Load the saved model, scaler, encoder, and sample data
with open(r'C:\Users\geeth\OneDrive\Desktop\GUVI Projects\CarDekho\Code files\CarDekhoPrice_Model.pkl', 'rb') as file:
    model = pickle.load(file)

with open(r'C:\Users\geeth\OneDrive\Desktop\GUVI Projects\CarDekho\Code files\CarDekhoPrice_Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open(r'C:\Users\geeth\OneDrive\Desktop\GUVI Projects\CarDekho\Code files\CarDekhoPrice_Encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Load reference data
reference_data = pd.read_excel(r'C:\Users\geeth\OneDrive\Desktop\GUVI Projects\CarDekho\Code files\FeaturesEngineered.xlsx')

# Define numerical and categorical columns
important_numerical_cols = ['Width', 'MaxPower', 'ManufactureYear', 'KilometersDriven', 'Length',
                            'WheelBase', 'KerbWeight', 'Torque', 'Engine', 'Mileage',
                            'PreviousOwners', 'Seats', 'Doors', 'Car_Age', 'TopSpeed']

important_categorical_cols = ['City', 'FuelType', 'BodyType', 'manufacturer', 'CarModel', 'Color',
                              'EngineType', 'TransmissionType', 'DriveType', 'FuelSupplySystem', 'RearBrakeType',
                              'TyreType', 'SteeringType', 'Locking', 'GearBox']

# Dynamic filtering based on manufacturer
def filter_data_by_manufacturer(reference_data, manufacturer):
    return reference_data[reference_data['manufacturer'] == manufacturer]

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
# Manufacturer selectbox (first input)
manufacturer = st.selectbox("Manufacturer", reference_data['manufacturer'].unique().tolist())

# Filter data dynamically based on manufacturer
filtered_data = filter_data_by_manufacturer(reference_data, manufacturer)

# Update options based on filtered data
cities = filtered_data['City'].unique().tolist()
fuel_types = filtered_data['FuelType'].unique().tolist()
body_types = filtered_data['BodyType'].unique().tolist()
car_models = filtered_data['CarModel'].unique().tolist()
colors = filtered_data['Color'].unique().tolist()
engine_types = filtered_data['EngineType'].unique().tolist()
transmission_types = filtered_data['TransmissionType'].unique().tolist()
drive_types = filtered_data['DriveType'].unique().tolist()
fuel_supply_systems = filtered_data['FuelSupplySystem'].unique().tolist()
rear_brake_types = filtered_data['RearBrakeType'].unique().tolist()
tyre_types = filtered_data['TyreType'].unique().tolist()
steering_types = filtered_data['SteeringType'].unique().tolist()
lockings = filtered_data['Locking'].unique().tolist()
gear_boxes = filtered_data['GearBox'].unique().tolist()

# Numerical inputs based on filtered data
numerical_ranges = {col: (filtered_data[col].min(), filtered_data[col].max()) for col in important_numerical_cols}

# Collect numerical inputs
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
seats = st.sidebar.selectbox("Seats", options=range(int(numerical_ranges['Seats'][0]), int(numerical_ranges['Seats'][1])+1))
doors = st.sidebar.slider("Doors", min_value=numerical_ranges['Doors'][0], max_value=numerical_ranges['Doors'][1], value=numerical_ranges['Doors'][0])
car_age = st.sidebar.slider("Car Age (years)", min_value=numerical_ranges['Car_Age'][0], max_value=numerical_ranges['Car_Age'][1], value=numerical_ranges['Car_Age'][0])
top_speed = st.sidebar.number_input("Top Speed (km/h)", min_value=numerical_ranges['TopSpeed'][0], max_value=numerical_ranges['TopSpeed'][1], value=numerical_ranges['TopSpeed'][0])

# Categorical inputs
city = st.selectbox("City", cities)
fuel_type = st.selectbox("Fuel Type", fuel_types)
body_type = st.selectbox("Body Type", body_types)
car_model = st.selectbox("Car Model", car_models)
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

# Collect the numerical and categorical data
numerical_data = np.array([[width, max_power, manufacture_year, kilometers_driven, length, 
                            wheel_base, kerb_weight, torque, engine, mileage, previous_owners,
                            seats, doors, car_age, top_speed]])

categorical_data = pd.DataFrame([[city, fuel_type, body_type, manufacturer, car_model, color, engine_type, 
                                  transmission_type, drive_type, fuel_supply_system, rear_brake_type, 
                                  tyre_type, steering_type, locking, gear_box]], 
                                columns=important_categorical_cols)

# Encode categorical data
encoded_data = encoder.transform(categorical_data).toarray()  # Convert sparse matrix to dense array

# Scale numerical data
scaled_data = scaler.transform(numerical_data)

# Ensure both arrays are 2D
numerical_data = np.atleast_2d(numerical_data)

# Check the shapes of both arrays
# st.write(f"Scaled Data Shape: {scaled_data.shape}")
# st.write(f"Encoded Data Shape: {encoded_data.shape}")

# Combine numerical and categorical data
final_data = np.hstack((scaled_data, encoded_data))

# Check the shape and content of final_data
# st.write(f"Final Data Shape: {final_data.shape}")
# st.write(f"Final Data Content: {final_data}")

# Prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(final_data)
        st.success(f"Predicted Car Price: ₹{prediction[0]:,.2f}Lakh")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
