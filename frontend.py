import streamlit as st
import pickle
import pandas as pd

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏠 Bangalore House Price Prediction")

sqft = st.number_input("Total Sqft", min_value=500, max_value=5000, step=10)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=5, step=1)
bhk = st.number_input("BHK", min_value=1, max_value=5, step=1)
price_per_sqft = st.number_input("Price per Sqft (₹)", min_value=1000, max_value=50000, step=100)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=3, step=1)
floor = st.number_input("Floor Number", min_value=0, max_value=30, step=1)
availability = st.selectbox("Availability Status", ["Not Ready", "Ready to Move"])

# Convert categorical input to numerical
availability = 1 if availability == "Ready to Move" else 0

if st.button("Predict Price"):
    features = pd.DataFrame([[sqft, bath, bhk, price_per_sqft, balcony, floor, availability]], 
                            columns=['total_sqft', 'bath', 'bhk', 'price_per_sqft', 'balcony', 'floor', 'availability'])
    prediction = model.predict(features)[0]
    st.success(f"🏡 Estimated Price: ₹{prediction:.2f} Lakhs")
