import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
try:
    trips_by_distance = pd.read_csv("Trips_By_Distance.csv")
    trips_full_data = pd.read_csv("Trips_Full_Data.csv")
except FileNotFoundError as e:
    print(f"Error: One or more data files not found. Please ensure 'Trips_By_Distance.csv' and 'Trips_Full_Data.csv' are in the same directory.\n{e}")
    exit()

# ---- 1. Calculate the Average Number of People Staying at Home per Week ----
if "Week" in trips_by_distance.columns and "Population Staying at Home" in trips_by_distance.columns:
    avg_people_staying_home = trips_by_distance.groupby("Week")["Population Staying at Home"].mean()
    overall_avg_staying_home = avg_people_staying_home.mean()
    print(f"Overall Weekly Average of People Staying at Home: {overall_avg_staying_home:,.0f}")
else:
    print("Error: Required columns missing in 'Trips_By_Distance.csv'.")

# ---- 2. Determine the Average Travel Distance for People Not Staying at Home ----
# Identify the correct column dynamically
travel_columns = [col for col in trips_full_data.columns if "Trips" in col and "Miles" in col]

# Fix missing "Week" column in Trips_Full_Data
if "Week of Date" in trips_full_data.columns:
    trips_full_data.rename(columns={"Week of Date": "Week"}, inplace=True)  # Rename for consistency

if "Week" in trips_full_data.columns and travel_columns:
    avg_travel_distance = trips_full_data.groupby("Week")[travel_columns].mean()
    overall_avg_travel = avg_travel_distance.mean().mean()
    print(f"Overall Weekly Average Travel Distance for Non-Home People: {overall_avg_travel:,.2f} miles")
else:
    print("Error: Required columns missing in 'Trips_Full_Data.csv'.")

# ---- 3. Identify Dates with Over 10,000,000 Trips ----
if "Date" in trips_by_distance.columns:
    threshold = 10_000_000
    if "Number of Trips 10-25" in trips_by_distance.columns:
        over_10M_10_25 = trips_by_distance[trips_by_distance["Number of Trips 10-25"] > threshold][["Date", "Number of Trips 10-25"]]
        print("\nDates with Over 10,000,000 Trips (10-25 Miles):")
        print(over_10M_10_25.head())

        # Scatter plot for trips 10-25
        plt.figure(figsize=(10,5))
        plt.scatter(over_10M_10_25["Date"], over_10M_10_25["Number of Trips 10-25"], color='blue', alpha=0.5)
        plt.xticks(rotation=45)
        plt.ylabel("Number of Trips (10-25 Miles)")
        plt.title("Dates with Over 10M Trips (10-25 Miles)")
        plt.grid()
        plt.show()

    if "Number of Trips 50-100" in trips_by_distance.columns:
        over_10M_50_100 = trips_by_distance[trips_by_distance["Number of Trips 50-100"] > threshold][["Date", "Number of Trips 50-100"]]
        print("\nDates with Over 10,000,000 Trips (50-100 Miles):")
        print(over_10M_50_100.head())

        # Scatter plot for trips 50-100
        plt.figure(figsize=(10,5))
        plt.scatter(over_10M_50_100["Date"], over_10M_50_100["Number of Trips 50-100"], color='red', alpha=0.5)
        plt.xticks(rotation=45)
        plt.ylabel("Number of Trips (50-100 Miles)")
        plt.title("Dates with Over 10M Trips (50-100 Miles)")
        plt.grid()
        plt.show()
else:
    print("Error: 'Date' column missing in 'Trips_By_Distance.csv'.")

