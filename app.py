import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI
import uvicorn

# Load Data
df = pd.read_csv("data/Banglore_House_Data.csv")

# Check column names
print("Columns in dataset:", df.columns)

# Convert total_sqft to numeric
def convert_sqft(x):
    try:
        return float(x)
    except:
        return np.mean([float(i) for i in x.split('-')]) if '-' in x else None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)

# Create 'bhk' if missing
if 'bhk' not in df.columns and 'size' in df.columns:
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if isinstance(x, str) else x)

# Convert 'floor' to numeric
if 'floor' in df.columns:
    df['floor'] = df['floor'].apply(lambda x: int(str(x).split(' ')[0]) if isinstance(x, str) and x[0].isdigit() else 0)
else:
    df['floor'] = 0  # Default value if 'floor' column is missing

# Feature Engineering
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

# Convert categorical 'availability' to binary (1 = Ready to move, 0 = Not Ready)
df['availability'] = df['availability'].apply(lambda x: 1 if x == 'Ready to move' else 0)

# Define Features & Target
features = ['total_sqft', 'bath', 'bhk', 'price_per_sqft', 'balcony', 'floor', 'availability']

# Ensure features exist in dataset
missing_features = [col for col in features if col not in df.columns]
if missing_features:
    print("Missing columns:", missing_features)
    exit()

# Remove rows with missing feature values
df = df.dropna(subset=features)

# Split Data
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")

# FastAPI for Predictions
app = FastAPI()

@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict/")
def predict_price(sqft: float, bath: int, bhk: int, price_per_sqft: float, balcony: int, floor: int, availability: int):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    features = pd.DataFrame([[sqft, bath, bhk, price_per_sqft, balcony, floor, availability]], 
                            columns=['total_sqft', 'bath', 'bhk', 'price_per_sqft', 'balcony', 'floor', 'availability'])
    
    prediction = model.predict(features)[0]
    return {"predicted_price": f"₹{prediction:.2f} Lakhs"}

if __name__ == "__main__":
    import os
    os.system("uvicorn app:app --host 0.0.0.0 --port 8000 --reload")

