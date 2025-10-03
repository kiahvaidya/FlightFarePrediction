# ✈️ Flight Fare Prediction

This project uses machine learning to predict flight fares based on user-provided flight details and suggests the best booking date to get the lowest fare.

## 🔧 Algorithms Used
- Random Forest Regressor ✅ (Best Performing)
- XGBoost Regressor

## 📊 Dataset
A sample flight dataset containing features like Airline, Source, Destination, Total Stops, Duration, and Price.

## 🧪 Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 🚀 How to Run

1. Clone the repo:

 git clone https://github.com/kiahvaidya/FlightFarePrediction.git
 cd flight-fare-prediction

2. Install dependencies:

  pip install -r requirements.txt

  Ensure rf_flight_model.xz is in the project directory.

  The app can automatically download it if missing.


## 🗒️ Notes

Predictions are based on historical data and the trained model; actual fares may vary.

Flight date must be today or a future date.

Only major Indian cities are supported.


