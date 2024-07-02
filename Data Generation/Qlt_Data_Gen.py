import pandas as pd
import numpy as np

# Load the initial data
initial_data = pd.DataFrame({
    'Date': ['2021-01-01', '2021-01-02'],
    'Equipment_ID': ['EQ-001', 'EQ-001'],
    'Total_Units_Produced': [1000, 1100],
    'Defective_Units': [50, 30]
})

# Extract Equipment ID from the initial data
equipment_id = initial_data.at[0, 'Equipment_ID']

# Number of rows to generate
num_rows = 900

# Generate synthetic data for Date column
start_date = pd.Timestamp(initial_data.at[1, 'Date']) + pd.Timedelta(days=1)
end_date = start_date + pd.Timedelta(days=num_rows - 1)
date_range = pd.date_range(start=start_date, end=end_date)

# Extract only the date component
date_range = date_range.date

# Generate synthetic data for Total Units Produced and Defective Units columns
total_units_produced = np.random.randint(100, 115, size=num_rows)*10
defective_units = np.random.randint(6, 10, size=num_rows)*5

# Create DataFrame with synthetic data
synthetic_data = pd.DataFrame({
    'Date': date_range,
    'Equipment_ID': [equipment_id] * num_rows,
    'Total_Units_Produced': total_units_produced,
    'Defective_Units': defective_units
})

# Concatenate initial data with synthetic data
final_data = pd.concat([initial_data, synthetic_data], ignore_index=True)


# Save synthetic data as Excel file
synthetic_data.to_excel("Qlt_data.xlsx", index=False)

# Save synthetic data as CSV file
synthetic_data.to_csv("Qlt_data.csv", index=False)