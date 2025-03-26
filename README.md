# 🏠 Bangalore House Price Prediction

This project predicts house prices in Bangalore using **Machine Learning** and serves predictions via a **FastAPI** backend.

## 📌 Features
✅ Cleans and preprocesses **Bangalore house price dataset**  
✅ **Trains a Linear Regression model** to predict house prices  
✅ **FastAPI-based REST API** for real-time price predictions  
✅ Supports **feature engineering** like price per sqft, floor, availability  
✅ API accessible via **Swagger UI (`/docs`)**  

---

## 📂 Project Structure

house_price_prediction/ │── data/ # Dataset folder │ ├── Banglore_House_Data.csv # Dataset file │── model.pkl # Saved machine learning model │── app.py # FastAPI backend & ML pipeline │── README.md # Project documentation │── requirements.txt # Python dependencies


---

## 📥 Dataset
The dataset used is **Bangalore House Price Data** from [Kaggle](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data).  
**Columns Used:**  
- `total_sqft` → Total area in square feet  
- `bath` → Number of bathrooms  
- `bhk` → Number of bedrooms  
- `balcony` → Number of balconies  
- `floor` → Floor number  
- `availability` → 1 if "Ready to move", else 0  
- `price` → House price (Target variable)  

---

## 🚀 Installation

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-repo/house_price_prediction.git
cd house_price_prediction
