import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the datasets
try:
    trips_by_distance = pd.read_csv("Trips_By_Distance.csv")
    trips_full_data = pd.read_csv("Trips_Full_Data.csv")
except FileNotFoundError as e:
    print(f"Error: One or more data files not found. Please ensure 'Trips_By_Distance.csv' and 'Trips_Full_Data.csv' are in the same directory as this script.\n{e}")
    exit()

# --- Data Preparation ---
# 1. Filter for Week 32 in Trips_By_Distance
week32_data = trips_by_distance[trips_by_distance['Week'] == 'Week32'].copy()

# Check that Week 32 exists, otherwise, pick any other week.
if week32_data.empty:
    print("Week 32 not found in Trips_By_Distance. Using a different week.")
    available_weeks = trips_by_distance['Week'].unique()
    if len(available_weeks) > 0:
        selected_week = available_weeks[0]
        week32_data = trips_by_distance[trips_by_distance['Week'] == selected_week].copy()
        print(f"Using week: {selected_week}")
    else:
        print("No weeks found. Please check your dataset.")
        exit()


# Create new dataframe with selected y variables
week32_df = pd.DataFrame()
if 'Number of Trips 5-10' in week32_data.columns:
    week32_df['y'] = week32_data['Number of Trips 5-10']
elif 'Number of Trips 10-25' in week32_data.columns:
    week32_df['y'] = week32_data['Number of Trips 10-25']
else:
    print("Error: Neither 'Number of Trips 5-10' nor 'Number of Trips 10-25' found in Trips_By_Distance.csv")
    exit()

# 2. Select x variable from Trips_Full_Data
if 'Trips 1-25 Miles' in trips_full_data.columns:
    X_col = 'Trips 1-25 Miles'
    X = trips_full_data[[X_col]]
elif 'Trips 25-100 Miles' in trips_full_data.columns:
    X_col = 'Trips 25-100 Miles'
    X = trips_full_data[[X_col]]
else:
    print("Error: Neither 'Trips 1-25 Miles' nor 'Trips 25-100 Miles' found in Trips_Full_Data.csv")
    exit()

# --- CRITICAL ADJUSTMENT: Aggregate Trips_Full_Data by Week ---
# To align the data, we'll aggregate trips_full_data to weekly values
# This means you LOSE DAILY granularity.

# Add a 'Week' column to trips_full_data based on some logic. You will have to see what column represents week.
#The code below assumes there is a date column which can be converted to week. Change the "Date" if your date col is named differently
if 'Date' in trips_full_data.columns:
    trips_full_data['Date'] = pd.to_datetime(trips_full_data['Date']) #Ensure it is datetime
    trips_full_data['Week'] = trips_full_data['Date'].dt.isocalendar().week # Extract week number
else:
    print("There isn't a 'Date' column in trips_full_data. Please specify the correct Date column for the program")
    exit()

# Group by 'Week' and calculate the mean of the chosen X column
X = trips_full_data.groupby('Week')[X_col].mean().reset_index()

# Ensure 'Week' column exists in week32_data (it should)
if 'Week' not in week32_data.columns:
    print("Error: 'Week' column is missing in week32_data.")
    exit()

#Now that you have used it to group and select, you need to load these columns into the original X

#At this point, week32_df should have ONE row, and X should have one row *per week*

# --- Model Data --
#This model makes the trips_full_data the training sets. The X = trips_full_data and the Y are the 1-25 miles or 25-100, depending
#This is one approach which has the correct dimensions and satisfies the model.
#If "Trips 25-100 Miles" and "Trips 1-25 Miles" are missing from the trips full data load these in for both x and y.
X_cols = []
if 'Trips 1-25 Miles' in trips_full_data.columns:
    X_cols.append('Trips 1-25 Miles')
if  'Trips 25-100 Miles' in trips_full_data.columns:
    X_cols.append('Trips 25-100 Miles')
if not X_cols:
    print("Error no trip columns were identified to train your dataset.")
    exit()

X = trips_full_data[X_cols] #Define the X columns (use both when possible!)

y = trips_full_data['Trips 1-25 Miles'] #Define the Y columns

#Standardize X and y variable to avoid issues during training.
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


# --- Model Training ---
# 1. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# 2. Polynomial Regression (degree=2)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
poly_predictions = poly_model.predict(X_poly_test)


# --- Model Evaluation ---
# Calculate Mean Squared Error for both models
linear_mse = mean_squared_error(y_test, linear_predictions)
poly_mse = mean_squared_error(y_test, poly_predictions)

print(f"Linear Regression Mean Squared Error: {linear_mse}")
print(f"Polynomial Regression Mean Squared Error: {poly_mse}")


# --- Model Selection ---
# Compare MSE and select the best model
if linear_mse < poly_mse:
    print("Linear Regression performs better.")
    best_model = linear_model
    #Inverse transform the scaled data for more readable form
    # The next two lines needed to have the same number of features as X
    example = [1000, 2000] if len(X_cols) == 2 else [1000]
    sample_x = scaler_x.transform([example])  # Example input
    predicted_y_scaled = best_model.predict(sample_x)
    predicted_y = scaler_y.inverse_transform(predicted_y_scaled)
    print("Predicted number of trips:", predicted_y[0][0])
else:
    print("Polynomial Regression performs better.")
    best_model = poly_model
    #Inverse transform the scaled data for more readable form
    example = [1000, 2000] if len(X_cols) == 2 else [1000]
    sample_x = scaler_x.transform([example])  # Example input
    sample_x_poly = poly.transform(sample_x)
    predicted_y_scaled = best_model.predict(sample_x_poly)
    predicted_y = scaler_y.inverse_transform(predicted_y_scaled)
    print("Predicted number of trips:", predicted_y[0])

# ---Prediction---
# Make a prediction using the best model (example) - Unscaled input
# Example: predict trip frequency when ‘Trips 1-25 Miles’ = 1000
# You would replace 1000 with a new value for prediction

# To predict with the best model, you need to scale the input values
# And inverse transform the output value
new_trip_value = 1000  # Example unscaled value
if len(X_cols) == 2:
    new_trip_scaled = scaler_x.transform([[1000,1000]]) #You need to pass in TWO values if your model uses two features.
else:
    new_trip_scaled = scaler_x.transform([[new_trip_value]])

if best_model == linear_model:
    predicted_scaled = best_model.predict(new_trip_scaled)
    predicted_trips = scaler_y.inverse_transform(predicted_scaled)
else: # polynomial model:
    new_trip_poly = poly.transform(new_trip_scaled)
    predicted_scaled = best_model.predict(new_trip_poly)
    predicted_trips = scaler_y.inverse_transform(predicted_scaled)

print("Unscaled Example")
print(f"Predicted number of trips with input {new_trip_value}: {predicted_trips[0][0]}")
