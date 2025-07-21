import pandas as pd

# Load datasets
train_path = '/mnt/data/extracted_dataset/Train.csv'
test_path = '/mnt/data/extracted_dataset/Test.csv'
variables_path = '/mnt/data/extracted_dataset/VariableDefinitions.csv'

# Read CSVs
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
variables_df = pd.read_csv(variables_path)

# Basic info
print("=== Train Dataset Info ===")
print(train_df.info())
print("\nTrain Head:")
print(train_df.head())

print("\n=== Test Dataset Info ===")
print(test_df.info())
print("\nTest Head:")
print(test_df.head())

print("\n=== Variable Definitions ===")
print(variables_df.head())

# Check for missing values
print("\nMissing values in Train:")
print(train_df.isnull().sum())

print("\nMissing values in Test:")
print(test_df.isnull().sum())

# Basic statistics
print("\nTrain Dataset Description:")
print(train_df.describe())