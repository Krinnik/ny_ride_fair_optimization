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

def impute_na_ratecodeID(ny_taxi_2024_df):
    def_mask = (ny_taxi_2024_df['PULocationID'] != ny_taxi_2024_df['PULocationID'].isin([np.isnan, 1, 132, 250, 265])) | (
        ny_taxi_2024_df['DOLocationID'] != ny_taxi_2024_df['DOLocationID'].isin([np.isnan, 1, 132, 250, 265])
    )
    rate_mask = (ny_taxi_2024_df['RatecodeID'] > 6) | np.isnan(ny_taxi_2024_df['RatecodeID'])
    mask = def_mask & rate_mask
    ny_taxi_2024_df.loc[mask, ['RatecodeID']] = 1


def impute_ratecodeIDs(ny_taxi_2024_df):
    mask = (ny_taxi_2024_df['PULocationID'] == 1) | (ny_taxi_2024_df['DOLocationID'] == 1)
    mask_2 = (ny_taxi_2024_df['PULocationID'] == 132) | (ny_taxi_2024_df['DOLocationID'] == 132)
    mask_3 = (ny_taxi_2024_df['PULocationID'] == 250) | (ny_taxi_2024_df['DOLocationID'] == 250) | (
        ny_taxi_2024_df['PULocationID'] == 265) | (ny_taxi_2024_df['DOLocationID'] == 265)
    ny_taxi_2024_df.loc[mask, ['RatecodeID']] = 3
    ny_taxi_2024_df.loc[mask_2, ['RatecodeID']] = 2
    ny_taxi_2024_df.loc[mask_3, ['RatecodeID']] = 4
    return ny_taxi_2024_df

def impute_negatives(ny_taxi_2024_df):
    negatives = ['fare_amount', 'extra', 'tip_amount', 'tolls_amount', 'mta_tax', 'improvement_surcharge', 'total_amount', 'congestion_surcharge']
    mask = (ny_taxi_2024_df[negatives] < 0).any(axis=1)
    mask_2 = (np.isnan(ny_taxi_2024_df['congestion_surcharge']))
    ny_taxi_2024_df.loc[mask, negatives] = ny_taxi_2024_df.loc[mask, negatives].abs()
    ny_taxi_2024_df.loc[mask_2, ['congestion_surcharge']] = 0
    return ny_taxi_2024_df