https://bikesshare.streamlit.app/

# 🚲 City Bike Sharing Demand Predictor

## Project Overview
This project predicts the hourly demand for city bike rentals based on weather conditions and human behavioral patterns. It includes an end-to-end machine learning pipeline and a fully functioning web application built with Streamlit.

## The Core Insight: Time-Series Feature Engineering
A raw timestamp ("2012-05-14 08:00:00") isn't very useful to a machine learning model. To capture complex human routines, I engineered specific behavioral features from the datetime data:
* `Is_Weekend`: Separating standard workweeks from weekend recreation.
* `Is_RushHour`: Identifying standard commuting windows (7-9 AM & 5-7 PM on weekdays).

## Methodology & Performance
* **Model:** Random Forest Regressor
* **Scaling:** Continuous weather variables (Temperature, Humidity, Windspeed) were standardized using `StandardScaler`.
* **Results:** The final model successfully explains over 90% of the variance in bike rentals (**R² > 0.90**), proving that time-of-day dictates base demand, while weather conditions act as the ultimate multipliers.

## Files in this Repository
* `bike_sharing_eda_and_training.ipynb`: The Jupyter Notebook containing data cleaning, Exploratory Data Analysis (EDA), model training, and validation.
* `app.py`: The Streamlit application script.
* `bike_rf_model.pkl` & `bike_scaler.pkl`: The saved Random Forest model and standard scaler required to run the app.
* `requirements.txt`: The Python dependencies needed to run the environment.

## How to Run the App Locally
If you want to run this Streamlit app on your local machine, run the following commands in your terminal:
```bash
git clone [https://github.com/walidkhaleed/bike-sharing-demand-app.git](https://github.com/walidkhaleed/bike-sharing-demand-app.git)
cd bike-sharing-demand-app
pip install -r requirements.txt
streamlit run app.py
