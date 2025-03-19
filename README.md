# MaxiMile

## Executive Summary  

The **MaxiMile** team has devise an ensemble model that can be used by For Hire Vehicle *(FHV - eg. Uber, Lyft, Bolt, etc)* riders predicts both trip duration and trip costs. Additionally, for FHV drivers on any platform, a model predicts potential hourly earnings and next location to maximize their earnings.

This initiative is led by three key team members:

- **Nik (Lead Software Engineer)**: Builds the backend infrastructure, including real-time weather/traffic integration, builds product logic demo.
- **Mike (Data Scientist)**: Develops the predictive models for fare, duration, and hourly earnings, feature engineering for model performance.
- **Zain (Product/Sales)**: Defined product vision and drives product/sales strategies, defined airport specific duration and cost model. 

## Data Landscape
The underlying data includes ride records, location coordinates, weather conditions, traffic levels, and timestamp information. 
We visualized this data to understand patterns and validate our models.

**Data Sources**
Below is the list of data sources that were incorporated into our model to make accurate predictions
- Trip Data | NYC Taxi and Limousine Commission (2024)
- Weather Data | Weather.com API, Metostat
- Geolocation Data | GeoPy

---

## Product Overview
Our prototype demonstrates basic predictions of:
- **Expected Trip Duration**
- **Expected Trip Fare**

The goal is to provide dynamic ride-hailing insights, integrating:
- **Real-time weather**
- **Traffic levels**
- **Date/time**

These features form a foundation for a user-facing UI that helps drivers identify high-earning locations and plan more efficient trips.

---

## Prototype Demo
- **Predictive Engine**  
  - Combines current location, destination, and dynamic external data.
- **Backend Infrastructure**  
  - Scalable and capable of real-time information updates (weather, traffic).
- **Demo**  
  - Early investor preview to gather feedback and demonstrate core functionality.
- **[Link to more detailed app README](streamlit_app/ny_ride_service_predictor/README.md)**
  - **Try it here: [MaxiMile](https://maximile.streamlit.app/)**
  
  <h1 align = "Left" > MaxiMile </h1>
  <h3 align = "left" > Rider App for New York City </h3>
  <p align = "left" >
  <img title="MaxiMile App" img src = "Resources/MaxiMile_App.png" alt = "MaxiMile App" width = "200"/>
    </p>

---

## Modeling Advances

### **Duration Model**
- Improved approach for accurately predicting trip times based on pickup/drop-off, congestion, etc.

### **Prototype Earnings-Per-Hour Model**
- Allows drivers to discover pick-up locations in NYC, where they might have a higher potential earnings.
- Outperforms a baseline (e.g., naive average predictions).

### **Future Modeling**
- Plans to refine feature engineering, add more robust historical data, and incorporate surge/peak events.
- Experiment with other models and methods to boost accuracy of predictions


---

## Next Steps

### **Integrate Nik’s Backend & Mike’s Models for a Highly Dynamic Prototype 2**
- Seamless connection between the predictive engine and the user-facing interface.

### **Secure Funding**
- Additional resources required to finalize and launch the product.

---

## Pilot Testing
- Early real-world trials to gather performance data, iterate on model improvements, and refine UI/UX.

---

## Expansion Data & Model Training
- Expand to include other major metropolitain areas with higher ridership density
- -- Boston, Chicago, Los Angeles, Miami, Philadelphia, San Francisco, Seattle, and Washington DC

---

## Contact & Further Information
For questions, demos, or investment opportunities, please contact:

Zain Master - zain.master@gmail.com
Michael Brady - mvbrady85@gmail.com
Nik Psyllas - nikpsyllas@gmail.com

