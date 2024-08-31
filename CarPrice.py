import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model, scaler, and encoder
with open('C:/Users/geeth/OneDrive/Desktop/GUVI Projects/CarDekho/Code files/car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('C:/Users/geeth/OneDrive/Desktop/GUVI Projects/CarDekho/Code files/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('C:/Users/geeth/OneDrive/Desktop/GUVI Projects/CarDekho/Code files/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Define all important features
important_numerical_cols = ['Width', 'MaxPower', 'ManufactureYear', 'KilometersDriven', 'Length',
                            'WheelBase', 'KerbWeight', 'Torque', 'Engine', 'Mileage',
                            'CargoVolume', 'PreviousOwners', 'Seats', 'Doors', 'Car_Age', 'TopSpeed']
important_categorical_cols = ['City', 'FuelType', 'BodyType', 'manufacturer', 'CarModel','Color',
                              'EngineType', 'TransmissionType', 'DriveType', 'FuelSupplySystem', 'RearBrakeType',
                              'TyreType', 'SteeringType', 'HeadLights', 'Locking', 'GearBox']

# Add CSS for the background image
page_bg_img = '''
<style>
body {
    background-image: url("C:/Users/geeth/OneDrive/Desktop/GUVI Projects/CarDekho/Code files/Car.jpg"); 
    background-size: cover
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to handle user inputs
def get_user_input():
    st.sidebar.header("Enter Car Features")

    # Numerical inputs as sliders on the sidebar
    numerical_features = {}
    for col in important_numerical_cols:
        min_val = 0.0  # Customize min value if needed
        max_val = 1000.0  # Customize max value if needed
        step = 0.1  # Customize step size if needed
        numerical_features[col] = st.sidebar.slider(f"{col}", min_value=min_val, max_value=max_val, value=0.0, step=step)

    # Categorical inputs as dropdowns/select boxes in the main body
    categorical_features = {}
    for col in important_categorical_cols:
        categorical_features[col] = st.selectbox(f"{col}", ["Option 1", "Option 2", "Option 3"])  # Replace with actual options

    # Combine into a DataFrame
    features = {**numerical_features, **categorical_features}
    features_df = pd.DataFrame(features, index=[0])
    return features_df

# Function to preprocess the input
def preprocess_input(features):
    # Separate numerical and categorical features
    numerical_features = features[important_numerical_cols]
    categorical_features = features[important_categorical_cols]

    # Handle missing numerical values by replacing them with the mean (or a specific strategy)
    numerical_features = numerical_features.fillna(numerical_features.mean())

    # Scale numerical features
    numerical_features_scaled = scaler.transform(numerical_features)

    # Handle missing categorical values by filling with 'unknown' or other default value
    categorical_features = categorical_features.fillna('unknown')

    # Encode categorical features
    try:
        categorical_features_encoded = encoder.transform(categorical_features).toarray()
    except ValueError as e:
        # If there's an encoding issue (e.g., unseen categories), handle it gracefully
        st.error(f"Encoding error: {e}")
        return None

    # Combine scaled and encoded features
    features_processed = np.hstack([numerical_features_scaled, categorical_features_encoded])

    return features_processed

# Predict function
def predict_price(features_processed):
    # Check if the processed features are valid
    if features_processed is not None:
        # Predict the price
        prediction = model.predict(features_processed)
        return prediction[0]
    else:
        return None

# Streamlit UI
st.title("Car Price Prediction App")
st.write("Enter the car features below and get an estimated price.")

# Get user input
user_input = get_user_input()

# Preprocess input and predict the price
if st.button("Predict Price"):
    features_processed = preprocess_input(user_input)
    if features_processed is not None:
        prediction = predict_price(features_processed)
        if prediction is not None:
            st.success(f"The estimated price of the car is: ${prediction:.2f}")
        else:
            st.error("Could not make a prediction based on the given inputs.")
    else:
        st.error("Failed to preprocess inputs. Please check the entered data.")




