# NY Ride Service Price and Duration Prediction

This Streamlit web application predicts the estimated price and travel duration for ride services (Yellow Cab, Uber, and Lyft) in New York City.
---

## ✅ Overview

The application allows users to:

* **Select Pickup and Dropoff Locations:** Choose from a dropdown list of 263 NYC ride zones.
* **Optionally Use an Interactive Map:** Click on markers on the map to automatically set the pickup or dropoff location.
* **Input Trip Time and Date:** Specify the desired time and date for the ride.
* **Get Prediction Results:** Upon clicking the "Run Prediction" button, the app displays estimated prices and durations for Yellow Cab, Uber, and Lyft.
* **View Trip Information:** Details such as the selected pickup and dropoff zones, estimated distance, current temperature at the pickup location, recent precipitation, and the requested time are shown.

---

## ✅ How It Works

The application utilizes machine learning models (trained on historical NYC taxi, uber, and lyft data) to generate the predictions. Key features considered by the models include:

* Time-based features (second of day, day of year, weekend, holiday, rush hours)
* Weather conditions (temperature, precipitation)
* Estimated travel distance
* Airport-related surcharges
* Congestion pricing
* Encoded pickup and dropoff location IDs (to capture location-specific price and trip duration variations)
* Vehicle class (to differentiate between Yellow Cab, Uber, and Lyft)

The user's input is processed to create a feature vector, which is then fed into the pre-trained machine learning models to obtain the price and duration predictions for each ride service.

---

## ✅ Usage

1.  Open the deployed Streamlit application in your web browser.
2.  Use the "Pickup Location" and "Dropoff Location" dropdown menus to select your desired starting and ending points.
3.  Alternatively, click on the blue taxi icons on the map to set the pickup or dropoff. The selected location will be reflected in the corresponding dropdown.
4.  Adjust the "Current Time" and "Current Date" in the sidebar to reflect the desired trip time.
5.  Click the "Run Prediction" button.
6.  The estimated prices and travel durations for Yellow Cab, Uber, and Lyft will be displayed below the map, along with detailed trip information.

---
### ✅ Try it here: [MaxiMile](https://maximile.streamlit.app/) ✅

--- 

## ❌ Disclaimer

The predictions provided by this application are estimates based on the underlying machine learning models and the data they were trained on. Actual prices and durations may vary depending on various real-world factors such as traffic conditions, surge pricing, route changes, and other unforeseen circumstances.
