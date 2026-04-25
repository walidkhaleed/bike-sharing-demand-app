import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model and scaler
model = joblib.load('bike_rf_model.pkl')
scaler = joblib.load('bike_scaler.pkl')

# Helper function for rush hour
def is_rush_hour(hour, is_weekend):
    if is_weekend == 0 and (7 <= hour <= 9 or 17 <= hour <= 19):
        return 1
    return 0

# 2. Build the Web App UI
st.title("🚲 City Bike Demand Predictor")
st.write("Adjust the sliders below to predict how many bikes will be rented during a specific hour.")

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Hour of the Day (0-23)", 0, 23, 8)
    day_of_week = st.selectbox("Day of the Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
    month = st.slider("Month (1-12)", 1, 12, 6)

with col2:
    temp_c = st.slider("Temperature (°C)", -10, 40, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    weather_sit = st.selectbox("Weather Condition", [1, 2, 3, 4], format_func=lambda x: ['Clear/Partly Cloudy', 'Misty/Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow'][x-1])

# 3. Process the Inputs
if st.button("Predict Bike Demand"):
    # Recreate the datetime logic
    is_weekend = 1 if day_of_week >= 5 else 0
    rush_hour = is_rush_hour(hour, is_weekend)
    norm_temp = temp_c / 41.0
    
    # Create the exact DataFrame structure the model expects
    input_data = pd.DataFrame({
        'season': [3],               
        'yr': [1],                   
        'mnth': [month],             
        'hr': [hour],                
        'holiday': [0],              
        'weekday': [day_of_week],    
        'workingday': [1 if is_weekend == 0 else 0],
        'weathersit': [weather_sit], 
        'temp': [norm_temp],         
        'atemp': [norm_temp],        
        'hum': [humidity / 100.0],   
        'windspeed': [0.2],          
        'Month': [month],            
        'DayOfWeek': [day_of_week],  
        'Year': [2012],              
        'Is_Weekend': [is_weekend],  
        'Is_RushHour': [rush_hour]   
    })
    
    # Scale the continuous variables
    columns_to_scale = ['temp', 'atemp', 'hum', 'windspeed']
    input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    st.success(f"🔥 Predicted Demand: **{int(prediction)} bikes**")
