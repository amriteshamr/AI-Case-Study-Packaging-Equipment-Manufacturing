import pandas as pd
import numpy as np

# Load the initial data
initial_data = pd.DataFrame({
    'Date': ['2021-01-01', '2021-01-02'],  # Corrected date format
    'Equipment_ID': ['EQ-001', 'EQ-001'],
    'Total_Available_Time (hours)': [720, 720],
    'Downtime (hours)': [60, 30]
})

# Extract Equipment ID and Total Available Time from the initial data
equipment_id = initial_data.at[0, 'Equipment_ID']
total_available_time = initial_data.at[0, 'Total_Available_Time (hours)']

# Number of rows to generate
num_rows = 900

# Generate synthetic data for Date column
start_date = pd.Timestamp(initial_data.at[0, 'Date']) + pd.Timedelta(days=1)  # Corrected index to 0
end_date = start_date + pd.Timedelta(days=num_rows - 1)
date_range = pd.date_range(start=start_date, end=end_date)

# Generate synthetic data for Downtime column
# Here, I'm assuming a maximum downtime of 60 minutes
downtime = np.random.randint(6, 13, size=num_rows) * 5

# Create DataFrame with synthetic data
synthetic_data = pd.DataFrame({
    'Date': date_range,
    'Equipment_ID': [equipment_id] * num_rows,
    'Total_Available_Time (hours)': [total_available_time] * num_rows,
    'Downtime (hours)': downtime
})

# Concatenate initial data with synthetic data
final_data = pd.concat([initial_data, synthetic_data], ignore_index=True)

# Save synthetic data as Excel file
synthetic_data.to_excel("Avail_data.xlsx", index=False)

# Save synthetic data as CSV file
synthetic_data.to_csv("Avail_data.csv", index=False)
