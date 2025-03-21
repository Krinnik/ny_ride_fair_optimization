import pandas as pd
import numpy as np
import geopandas as gpd
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from shapely import wkb
from shapely import errors
import itertools
import matplotlib.pyplot as plt
import datetime as dt
import holidays
import math

# Imputation Functions

def impute_negatives(ny_taxi_2024_df):
    ny_taxi_2024_df = ny_taxi_2024_df.dropna()
    negatives = ['fare_amount', 'extra', 'tip_amount', 'tolls_amount', 'mta_tax', 'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'Airport_fee']
    mask = (ny_taxi_2024_df[negatives] < 0).any(axis=1)
    ny_taxi_2024_df = ny_taxi_2024_df.drop(ny_taxi_2024_df[mask].index)
    
    return ny_taxi_2024_df

def impute_airport_fee(ny_taxi_2024_df):
    
    pu_mask = (ny_taxi_2024_df['PULocationID'] == 132) | (ny_taxi_2024_df['PULocationID'] == 138)
    fee_mask = (ny_taxi_2024_df['Airport_fee'] == 1.25) | (ny_taxi_2024_df['Airport_fee'] == 0)
    mask = pu_mask & fee_mask
    ny_taxi_2024_df.loc[mask, 'Airport_fee'] = 1.75
    return ny_taxi_2024_df

def impute_outliers_airport_fee(ny_taxi_2024_df):
    
    pu_mask = (
        ny_taxi_2024_df['PULocationID'] != 132) & (
        ny_taxi_2024_df['PULocationID'] != 138)
    fee_mask = (
        ny_taxi_2024_df['Airport_fee'] == 1.25) | (ny_taxi_2024_df['Airport_fee'] == 1.75)
    mask = pu_mask & fee_mask
    ny_taxi_2024_df.loc[mask, 'Airport_fee'] = 0
    return ny_taxi_2024_df

def trip_distance_weird_maxes(ny_taxi_2024_df):
    long = (ny_taxi_2024_df['trip_distance'] > 75)
    ny_taxi_2024_df = ny_taxi_2024_df.drop(ny_taxi_2024_df[long].index)
    return ny_taxi_2024_df

def drop_unknowns(ny_taxi_2024_df):
    drop_rows = (ny_taxi_2024_df['PULocationID'] == 265) | (ny_taxi_2024_df['DOLocationID'] == 265) | (
        ny_taxi_2024_df['PULocationID'] == 264) | (ny_taxi_2024_df['DOLocationID'] == 264
    ) | (ny_taxi_2024_df['payment_type'] == 3) | (ny_taxi_2024_df['payment_type'] == 5) | (
        ny_taxi_2024_df['payment_type'] == 6 
    )
    ny_taxi_2024_df = ny_taxi_2024_df.drop(ny_taxi_2024_df[drop_rows].index)
    return ny_taxi_2024_df

def fix_total_amount(ny_taxi_2024_df):
    fix_1 = (ny_taxi_2024_df['fare_amount'] > ny_taxi_2024_df['total_amount'])
    fix_2 = (ny_taxi_2024_df['fare_amount'] < ny_taxi_2024_df['total_amount'])
    columns_to_sum = [
        'fare_amount',
        'extra',
        'mta_tax',
        'tip_amount',
        'tolls_amount',
        'improvement_surcharge',
        'congestion_surcharge',
        'Airport_fee'
    ]
    sum_of_columns = ny_taxi_2024_df[columns_to_sum].sum(axis=1)

    fix_amount = fix_1 & fix_2
    
    ny_taxi_2024_df.loc[fix_amount, ['total_amount']] = sum_of_columns[fix_amount]
    return ny_taxi_2024_df


def clean_taxi_data(ny_taxi_2024_df):
    ny_taxi_2024_df['service'] = 0
    ny_taxi_2024_df = ny_taxi_2024_df[['service',
                                       'tpep_pickup_datetime',
                                       'tpep_dropoff_datetime',
                                       'PULocationID',
                                       'DOLocationID',
                                       'trip_distance',
                                       'fare_amount',
                                       'tolls_amount',
                                       'congestion_surcharge',
                                       'Airport_fee',
                                       'total_amount']]
    ny_taxi_2024_df = ny_taxi_2024_df.sample(frac=0.08, random_state=29)
    return ny_taxi_2024_df



def clean_fhv_df(fhv_2024_df):
    fhv_2024_df = fhv_2024_df[['hvfhs_license_num', 'pickup_datetime',
       'dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_miles', 'base_passenger_fare', 'tolls',
       'congestion_surcharge', 'airport_fee']]
    
    fhv_2024_df.columns = ['service', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_distance', 'fare_amount', 'tolls_amount',
       'congestion_surcharge', 'Airport_fee']
    
    fhv_2024_df['trip_distance'] = round(fhv_2024_df['trip_distance'], 2)
    
    return fhv_2024_df

def impute_service(fhv_2024_df):
    uber = fhv_2024_df['service'] == 'HV0003'
    lift = fhv_2024_df['service'] == 'HV0005'


    fhv_2024_df.loc[uber, 'service'] = 1
    fhv_2024_df.loc[lift, 'service'] = 2

    return fhv_2024_df

def resample_uber(fhv_2024_df):
    uber = fhv_2024_df['service'] == 1
    uber_rows = fhv_2024_df[uber]
    lift = fhv_2024_df['service'] == 2
    lift_rows = fhv_2024_df[lift]
    ubers = uber_rows.sample(frac=0.33, random_state=29)

    fhv_2024_df = pd.concat([ubers, lift_rows])

    return fhv_2024_df

def impute_negatives_fhv(fhv_2024_df):
    
    columns = ['fare_amount', 'tolls_amount', 'congestion_surcharge', 'Airport_fee']
    neg = (fhv_2024_df[columns] < 0).any(axis=1)
    fhv_2024_df = fhv_2024_df.drop(fhv_2024_df[neg].index)

    return fhv_2024_df

def impute_fhv_airport_fee(fhv_2024_df):
    ids = [1, 132, 138]
    airport_yes = (fhv_2024_df['PULocationID'].isin(ids)) | (fhv_2024_df['DOLocationID'].isin(ids))
    airport_no = (~fhv_2024_df['PULocationID'].isin(ids)) & (~fhv_2024_df['DOLocationID'].isin(ids))

    fhv_2024_df.loc[airport_yes, 'Airport_fee'] = 2.50
    fhv_2024_df.loc[airport_no, 'Airport_fee'] = 0
    return fhv_2024_df

def calc_total(fhv_2024_df):
    calc = (fhv_2024_df.index)
    fhv_2024_df['total_amount'] = 0
    columns_to_sum = [
        'fare_amount', 'tolls_amount', 'congestion_surcharge', 'Airport_fee'
    ]

    sum_of_columns = fhv_2024_df[columns_to_sum].sum(axis=1)

    fhv_2024_df.loc[calc, 'total_amount'] = sum_of_columns[calc]

    return fhv_2024_df


def merge_data(ny_taxi_2024_df, fhv_2024_df):
    fhv_2024_df = fhv_2024_df.reset_index()
    fhv_2024_df = fhv_2024_df.drop('index', axis=1)
    merged_2024_df = pd.concat([ny_taxi_2024_df, fhv_2024_df])

    merged_2024_df = merged_2024_df[merged_2024_df['tpep_pickup_datetime'].dt.year == 2024]
    merged_2024_df = merged_2024_df[merged_2024_df['tpep_dropoff_datetime'].dt.year == 2024]

    merged_2024_df = merged_2024_df.sort_values('tpep_pickup_datetime', ascending=True)

    return merged_2024_df


def drop_unknowns_fhv(merged_2024_df):
    drop_rows = (merged_2024_df['PULocationID'] == 265) | (merged_2024_df['DOLocationID'] == 265) | (
        merged_2024_df['PULocationID'] == 264) | (merged_2024_df['DOLocationID'] == 264)
    merged_2024_df = merged_2024_df.drop(merged_2024_df[drop_rows].index)

    return merged_2024_df

def drop_high_fare(merged_2024_df):
    high = merged_2024_df['fare_amount'] > 500
    merged_2024_df = merged_2024_df.drop(merged_2024_df[high].index)

    return merged_2024_df

def get_times(merged_2024_df):
    merged_2024_df['ride_length'] = (merged_2024_df['tpep_dropoff_datetime'] - merged_2024_df['tpep_pickup_datetime']).dt.total_seconds()
    merged_2024_df['second_of_day'] = (merged_2024_df['tpep_pickup_datetime'].dt.hour * 3600 + merged_2024_df['tpep_pickup_datetime'].dt.minute * 60 + merged_2024_df['tpep_pickup_datetime'].dt.second)
    merged_2024_df['day_of_year'] = merged_2024_df['tpep_pickup_datetime'].dt.day_of_year
    merged_2024_df['is_weekend'] = merged_2024_df['tpep_pickup_datetime'].dt.weekday >= 5
    us_holidays = holidays.US()
    merged_2024_df['is_holiday'] = merged_2024_df['tpep_pickup_datetime'].apply(lambda x: 1 if x.date() in us_holidays else 0)
    merged_2024_df['hour'] = merged_2024_df['hour'] = merged_2024_df['second_of_day'].apply(lambda x: math.floor(x / 60 / 60))

    return merged_2024_df

def impute_geo_data(merged_2024_df, zone_long_lat_data):

    merged_2024_df["morning_rush_hour"] = ((merged_2024_df["tpep_pickup_datetime"].dt.weekday < 5) & 
                           (merged_2024_df["tpep_pickup_datetime"].dt.hour.between(7, 9))).astype(int)
    merged_2024_df["evening_rush_hour"] = ((merged_2024_df["tpep_pickup_datetime"].dt.weekday < 5) & 
                            (merged_2024_df["tpep_pickup_datetime"].dt.hour.between(16, 18))).astype(int)
    return merged_2024_df
    
def impute_weather(merged_2024_df, weather_df):
    merged_2024_df = merged_2024_df.dropna()
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df = weather_df.rename(columns={'time': 'tpep_pickup_datetime'})
    weather_df['hour'] = (weather_df['tpep_pickup_datetime']).dt.hour.astype('int')
    weather_df['day_of_year'] = (weather_df['tpep_pickup_datetime']).dt.day_of_year.astype('int')

    weather_df_clean = weather_df[['day_of_year', 'hour', 'prcp', 'temp(f)']]

    
    merged_2024_df = pd.merge(merged_2024_df, weather_df_clean[['day_of_year', 'hour', 'prcp', 'temp(f)']], on=['day_of_year', 'hour'], how='left')

    merged_2024_df = merged_2024_df.drop('hour', axis=1)

    return merged_2024_df


def clean_merged(merged_2024_df):
    merged_2024_df = merged_2024_df[['second_of_day', 'day_of_year', 'is_weekend', 'is_holiday', 'morning_rush_hour', 'evening_rush_hour', 'PULocationID', 'DOLocationID', 'prcp', 'temp(f)', 'trip_distance', 'ride_length', 'fare_amount', 'tolls_amount', 'Airport_fee', 'congestion_surcharge', 'total_amount', 'service']]
    
    merged_2024_df.columns = [['second_of_day', 'day_of_year', 'weekend', 'holiday', 'morning_rush', 'evening rush', 'PULocationID', 'DOLocationID', 'prcp', 'temp', 'distance', 'duration(sec)', 'fare', 'tolls', 'airport', 'congestion', 'total', 'class']]
    merged_2024_df = merged_2024_df.dropna()
    return merged_2024_df


def impute_all(ny_taxi_2024_df, fhv_2024_df, zone_long_lat_data, weather_df):
    ny_taxi_2024_df = impute_negatives(ny_taxi_2024_df)
    ny_taxi_2024_df = impute_airport_fee(ny_taxi_2024_df)
    ny_taxi_2024_df = impute_outliers_airport_fee(ny_taxi_2024_df)
    ny_taxi_2024_df = trip_distance_weird_maxes(ny_taxi_2024_df)
    ny_taxi_2024_df = drop_unknowns(ny_taxi_2024_df)
    ny_taxi_2024_df = fix_total_amount(ny_taxi_2024_df)
    ny_taxi_2024_df = clean_taxi_data(ny_taxi_2024_df)
    fhv_2024_df = clean_fhv_df(fhv_2024_df)
    fhv_2024_df = impute_service(fhv_2024_df)
    fhv_2024_df = resample_uber(fhv_2024_df)
    fhv_2024_df = impute_negatives_fhv(fhv_2024_df)
    fhv_2024_df = impute_fhv_airport_fee(fhv_2024_df)
    fhv_2024_df = calc_total(fhv_2024_df)
    merged_2024_df = merge_data(ny_taxi_2024_df, fhv_2024_df)
    merged_2024_df = drop_unknowns_fhv(merged_2024_df)
    merged_2024_df = drop_high_fare(merged_2024_df)
    merged_2024_df = get_times(merged_2024_df)
    merged_2024_df = impute_geo_data(merged_2024_df, zone_long_lat_data)
    merged_2024_df = impute_weather(merged_2024_df, weather_df)
    merged_2024_df = clean_merged(merged_2024_df)

    return merged_2024_df