import os
import pandas as pd

# Load CSV file
df = pd.read_csv('data.csv')

print("Original Data:")
print(df.head())

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Remove leading/trailing whitespace and convert to lowercase
df = df.map(lambda x: x.strip().lower() if isinstance(x, str) else x)

# Drop rows with any missing values
df.dropna(inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Create output directory if it doesn't exist
output_dir = 'ModelCleaning'
os.makedirs(output_dir, exist_ok=True)

# Save cleaned data
output_path = os.path.join(output_dir, 'cleaned_data.csv')
df.to_csv(output_path, index=False)

print("\nCleaned Data:")
print(df.head())
print(f"\nCleaned data saved to '{output_path}'")