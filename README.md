# MaxiMile

## Executive Summary  

The **MaxiMile** team has devise an ensemble model that can be used by For Hire Vehicle *(FHV - eg. Uber, Lyft, Bolt, etc)* riders predicts both trip duration and trip costs. Additionally, for FHV drivers on any platform, a model predicts potential hourly earnings and next location to maximize their earnings.

This initiative is led by three key team members:

- **Zain (Product & Sales)**: Defines product vision and drives sales strategies, bridging technical and non-technical stakeholders.
- **Mike (Data Scientist)**: Develops the predictive models for fare, duration, and hourly earnings, feature engineering for model performance.
- **Nik (Lead Software Engineer)**: Builds the backend infrastructure, including real-time weather/traffic integration, builds product logic demo.

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
- **Fare**
- **Trip duration**

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

---

## Modeling Advances

### **Duration Model**
- Improved approach for accurately predicting trip times based on pickup/drop-off, congestion, etc.

### **Prototype Earnings-Per-Hour Model**
- Allows drivers to see where they might earn more.
- Outperforms a baseline (e.g., naive average predictions).

### **Future Modeling**
- Plans to refine feature engineering, add more robust historical data, and incorporate surge/peak events.

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

## Contact & Further Information
For questions, demos, or investment opportunities, please contact:

Michael Brady - mvbrady85@gmail.com
Nik Psyllas - nikpsyllas@gmail.com
Zain Master - zain.master@gmail.com
