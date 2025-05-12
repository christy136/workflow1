import os
from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Path to cleaned data
csv_file_path = 'ModelCleaning/cleaned_data.csv'

# Confirm file exists
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"File not found at: {csv_file_path}")

# Read cleaned CSV
df = read_csv(csv_file_path)
print("Cleaned Data Preview:")
print(df.head())

# Prepare features and target
X = df["Age"].astype(float).values.reshape(-1, 1)
y = df["Salary"].astype(float)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model to file
model_path = 'ModelCleaning/AgeSalaryModel.pkl'
dump(model, model_path)

print(f"\nModel trained and saved to '{model_path}'")