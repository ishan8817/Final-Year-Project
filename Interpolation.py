import pandas as pd
from scipy.interpolate import interp1d

# Load datasets
core19_df = pd.read_csv("Core19_IPC.csv")
core3_df = pd.read_csv("Core3_IPC.csv")

# Merge datasets based on Cumulative Instructions column
merged_df = pd.merge(core19_df, core3_df, on="Cumulative Instructions", how="left", suffixes=('_19', '_03'))

# Perform linear interpolation to fill missing values in IPC_03 column
interp_func = interp1d(core3_df['Cumulative Instructions'], core3_df['IPC_03'], kind='linear', fill_value='extrapolate')

# Perform interpolation only for Cumulative Instructions within the range of Core19_IPC.csv
merged_df['IPC_03'] = interp_func(merged_df['Cumulative Instructions'])

# Reorder columns
merged_df = merged_df[['Cumulative Instructions', 'IPC_19', 'IPC_03']]

# Round IPC_19 and IPC_03 values to four decimal places
merged_df['IPC_19'] = merged_df['IPC_19'].round(4)
merged_df['IPC_03'] = merged_df['IPC_03'].round(4)

# Convert Cumulative Instructions column to integer type to remove decimal part if it's ".0"
merged_df['Cumulative Instructions'] = merged_df['Cumulative Instructions'].astype(int)

# Drop duplicate rows
merged_df.drop_duplicates(inplace=True)

# Write merged data to a new CSV file
merged_df.to_csv("Merged_IPC_6.csv", index=False)
