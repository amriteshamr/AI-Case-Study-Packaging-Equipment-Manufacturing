import pandas as pd
import numpy as np

# Load the initial data
initial_data = pd.DataFrame({
    'Date': ['2021-01-01', '2021-01-02'],
    'Equipment_ID': ['EQ-001', 'EQ-001'],
    'Ideal_Cycle_Time (seconds)': [60, 60],
    'Actual_Production_Time (hours)': [650, 600]
})

# Extract Equipment ID and Ideal Cycle Time from the initial data
equipment_id = initial_data.at[0, 'Equipment_ID']
ideal_cycle_time = initial_data.at[0, 'Ideal_Cycle_Time (seconds)']

# Number of rows to generate
num_rows = 900

# Generate synthetic data for Date column
start_date = pd.Timestamp(initial_data.at[1, 'Date']) + pd.Timedelta(days=1)
end_date = start_date + pd.Timedelta(days=num_rows - 1)
date_range = pd.date_range(start=start_date, end=end_date)

# Generate synthetic data for Actual Production Time column
# Here, I'm assuming a maximum actual production time of 720 hours
actual_production_time = np.random.randint(120, 130, size=num_rows)*5

# Create DataFrame with synthetic data
synthetic_data = pd.DataFrame({
    'Date': date_range,
    'Equipment_ID': [equipment_id] * num_rows,
    'Ideal_Cycle_Time (seconds)': [ideal_cycle_time] * num_rows,
    'Actual_Production_Time (hours)': actual_production_time
})

# Concatenate initial data with synthetic data
final_data = pd.concat([initial_data, synthetic_data], ignore_index=True)


# Save synthetic data as Excel file
synthetic_data.to_excel("Perfo_data.xlsx", index=False)

# Save synthetic data as CSV file
synthetic_data.to_csv("Perfo_data.csv", index=False)