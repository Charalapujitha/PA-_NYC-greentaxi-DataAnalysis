import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_parquet("green_tripdata_2022-02.parquet")

# Feature engineering
df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour

# Drop column if exists
df = df.drop(columns=['ehail_fee'], errors='ignore')

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=[
    'store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'weekday', 'hourofday'
])

# Feature selection
features = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
            'improvement_surcharge', 'congestion_surcharge', 'trip_duration', 'passenger_count'] + \
           [col for col in df_encoded.columns if col.startswith(('store_and_fwd_flag_', 'RatecodeID_', 'payment_type_', 'trip_type_', 'weekday_', 'hourofday_'))]

X = df_encoded[features].fillna(0)
y = df_encoded['total_amount'].fillna(0)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🚕 NYC Green Taxi Fare Prediction")
st.write("Predict the total fare amount of NYC Green Taxi rides using a regression model.")

# User inputs
user_input = {}

user_input['trip_distance'] = st.slider("Trip Distance (miles)", 0.0, 30.0, 2.5)
user_input['fare_amount'] = st.slider("Base Fare Amount ($)", 0.0, 50.0, 10.0)
user_input['extra'] = st.slider("Extra Charges ($)", 0.0, 10.0, 1.0)
user_input['mta_tax'] = st.selectbox("MTA Tax", [0.0, 0.5])
user_input['tip_amount'] = st.slider("Tip Amount ($)", 0.0, 20.0, 3.0)
user_input['tolls_amount'] = st.slider("Tolls Amount ($)", 0.0, 20.0, 0.0)
user_input['improvement_surcharge'] = st.selectbox("Improvement Surcharge", [0.0, 0.3])
user_input['congestion_surcharge'] = st.selectbox("Congestion Surcharge", [0.0, 2.5])
user_input['trip_duration'] = st.slider("Trip Duration (minutes)", 1, 120, 15)
user_input['passenger_count'] = st.slider("Passenger Count", 1, 6, 1)

# Encode categorical features
categorical_cols = {
    'store_and_fwd_flag': ['N', 'Y'],
    'RatecodeID': df['RatecodeID'].unique(),
    'payment_type': df['payment_type'].unique(),
    'trip_type': df['trip_type'].unique(),
    'weekday': df['weekday'].unique(),
    'hourofday': list(range(24))
}

for col, options in categorical_cols.items():
    choice = st.selectbox(f"{col}", sorted(options))
    for option in options:
        col_name = f"{col}_{option}"
        user_input[col_name] = 1 if choice == option else 0

# Create input dataframe
input_df = pd.DataFrame([user_input])

# Align input with training data
missing_cols = set(X.columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[X.columns]

# Prediction
prediction = model.predict(input_df)[0]
st.subheader(f"💵 Predicted Total Fare: ${prediction:.2f}")
