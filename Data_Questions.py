import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ---- 1. Load datasets efficiently ----
try:
    trips_by_distance = dd.read_csv(
        "Trips_By_Distance.csv",
        assume_missing=True,
        dtype={"County Name": "object", "Week": "object", "State Postal Code": "object"},
        usecols=["Week", "Population Staying at Home", "Date", "Number of Trips 10-25", "Number of Trips 50-100"],
        low_memory=False
    )

    trips_full_data = dd.read_csv(
        "Trips_Full_Data.csv",
        assume_missing=True,
        dtype={"Week of Date": "object", "Month of Date": "object"},
        usecols=["Week of Date", "Trips 1-25 Miles", "Trips 10-25 Miles", "Trips 50-100 Miles"],
        low_memory=False
    )
except FileNotFoundError as e:
    print(f"Error: One or more data files not found. Ensure they are in the same directory.\n{e}")
    exit()

# ---- 2. Calculate the Average Number of People Staying at Home per Week ----
if "Week" in trips_by_distance.columns and "Population Staying at Home" in trips_by_distance.columns:
    avg_people_staying_home = trips_by_distance.groupby("Week")["Population Staying at Home"].mean().compute()
    overall_avg_staying_home = avg_people_staying_home.mean()
    print(f"Overall Weekly Average of People Staying at Home: {overall_avg_staying_home:,.0f}")
else:
    print("Error: Required columns missing in 'Trips_By_Distance.csv'.")

# ---- 3. Determine the Average Travel Distance for People Not Staying at Home ----
if "Week of Date" in trips_full_data.columns:
    trips_full_data = trips_full_data.rename(columns={"Week of Date": "Week"})

travel_columns = [col for col in trips_full_data.columns if "Trips" in col and "Miles" in col]

if "Week" in trips_full_data.columns and travel_columns:
    avg_travel_distance = trips_full_data.groupby("Week")[travel_columns].mean().compute()
    overall_avg_travel = avg_travel_distance.mean().mean()
    print(f"Overall Weekly Average Travel Distance for Non-Home People: {overall_avg_travel:,.2f} miles")
else:
    print("Error: Required columns missing in 'Trips_Full_Data.csv'.")

# ---- 4. Identify Dates with Over 10,000,000 Trips ----
if "Date" in trips_by_distance.columns:
    threshold = 10_000_000

    def filter_large_trips(column):
        return trips_by_distance[trips_by_distance[column] > threshold][["Date", column]].compute()

    trip_columns = ["Number of Trips 10-25", "Number of Trips 50-100"]

    # Reduce parallel jobs for large datasets
    over_10M_results = Parallel(n_jobs=2)(
        delayed(filter_large_trips)(col) for col in trip_columns if col in trips_by_distance.columns
    )

    for result, col in zip(over_10M_results, trip_columns):
        if not result.empty:
            print(f"\nDates with Over 10,000,000 Trips ({col}):")
            print(result.head())

            # Scatter plot for large trips
            plt.figure(figsize=(10, 5))
            plt.scatter(result["Date"], result[col], alpha=0.5)
            plt.xticks(rotation=45)
            plt.ylabel(f"Number of Trips ({col})")
            plt.title(f"Dates with Over 10M Trips ({col})")
            plt.grid()
            plt.show()
else:
    print("Error: 'Date' column missing in 'Trips_By_Distance.csv'.")
