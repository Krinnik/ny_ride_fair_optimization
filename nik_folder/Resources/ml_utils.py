import pandas as pd
import numpy as np
import geopandas as gp
from geopy.geocoders import Nominatim


# Imputation Functions

def impute_passenger_count(ny_taxi_2024_df):
    ny_taxi_2024_df['passenger_count'] = ny_taxi_2024_df['passenger_count'].fillna(1)
    pass_mask = (ny_taxi_2024_df['passenger_count'] == 0)

    ny_taxi_2024_df.loc[pass_mask, 'passenger_count'] = 1

    return ny_taxi_2024_df



def impute_airport_fee(ny_taxi_2024_df):
    
    pu_mask = (ny_taxi_2024_df['PULocationID'] == 132) | (ny_taxi_2024_df['PULocationID'] == 138)
    fee_mask = np.isnan(ny_taxi_2024_df['Airport_fee']) | (ny_taxi_2024_df['Airport_fee'] == 1.25) | (ny_taxi_2024_df['Airport_fee'] == -1.75)
    mask = pu_mask & fee_mask
    ny_taxi_2024_df.loc[mask, 'Airport_fee'] = 1.75
    return ny_taxi_2024_df



def impute_outliers_airport_fee(ny_taxi_2024_df):
    
    pu_mask = (
        ny_taxi_2024_df['PULocationID'] != 132) | (
        ny_taxi_2024_df['PULocationID'] != 138) | (
            ny_taxi_2024_df['DOLocationID'] != 132) | (ny_taxi_2024_df['DOLocationID'] != 138)
    fee_mask = np.isnan(
        ny_taxi_2024_df['Airport_fee']) | (
        ny_taxi_2024_df['Airport_fee'] == 1.25) | (
            ny_taxi_2024_df['Airport_fee'] == -1.75)
    mask = pu_mask & fee_mask
    ny_taxi_2024_df.loc[mask, 'Airport_fee'] = 0
    return ny_taxi_2024_df