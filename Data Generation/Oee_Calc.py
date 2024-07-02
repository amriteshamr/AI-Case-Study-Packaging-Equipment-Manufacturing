import pandas as pd

# Load the three datasets
dataset1 = pd.read_csv("Avail_data.csv")
dataset2 = pd.read_csv("Perfo_data.csv")
dataset3 = pd.read_csv("Qlt_data.csv")

# Concatenate the datasets
all_datasets = pd.concat([dataset1, dataset2, dataset3])

grouped_data = all_datasets.groupby(['Date', 'Equipment_ID']).agg({
    'Total_Available_Time (hours)': 'sum',
    'Actual_Production_Time (hours)': 'sum',
    'Total_Units_Produced': 'sum',
    'Defective_Units': 'sum'
}).reset_index()

grouped_data['Ideal_Cycle_Time (seconds)'] = grouped_data['Total_Available_Time (hours)'] * 3600 / grouped_data['Total_Units_Produced']
grouped_data['Good_Units_Produced'] = grouped_data['Total_Units_Produced'] - grouped_data['Defective_Units']

grouped_data['Availability'] = grouped_data['Actual_Production_Time (hours)'] / grouped_data['Total_Available_Time (hours)']
grouped_data['Performance'] = grouped_data['Total_Units_Produced'] / (grouped_data['Total_Available_Time (hours)'] / 24 * 60 * 60 / grouped_data['Ideal_Cycle_Time (seconds)'])
grouped_data['Quality'] = grouped_data['Good_Units_Produced'] / grouped_data['Total_Units_Produced']

# Cap values to ensure they are not greater than 1
grouped_data['Availability'] = grouped_data['Availability'].clip(upper=1.0)
grouped_data['Performance'] = grouped_data['Performance'].clip(upper=1.0)
grouped_data['Quality'] = grouped_data['Quality'].clip(upper=1.0)

grouped_data['OEE'] = (grouped_data['Availability'] * grouped_data['Performance'] * grouped_data['Quality'] * 100).round(2) 

# Save the results as Excel and CSV files
grouped_data[['Date', 'Equipment_ID', 'OEE']].to_excel("oee_results.xlsx", index=False)
grouped_data[['Date', 'Equipment_ID', 'OEE']].to_csv("oee_results.csv", index=False)

print("OEE results saved as 'oee_results.xlsx' and 'oee_results.csv'.")