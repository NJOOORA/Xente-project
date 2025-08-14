# Xente Credit Default Prediction Project

This project predicts loan default risk using machine learning models on the Xente dataset.

## Project Structure

- `Xente project.py` – Main Streamlit app and model code
- `Train.csv` – Training dataset
- `Test.csv` – Test dataset
- `requirements.txt` – Python dependencies
- `README.md` – Project documentation

## Getting Started

### 1. Install Requirements

```sh
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```sh
streamlit run "Xente project.py"
```

### 3. Project Files

- **Train.csv**: Contains historical loan data and default labels.
- **Test.csv**: Contains new loan data for prediction.
- **requirements.txt**: Lists all required Python packages.

## Example Usage

```python
import pandas as pd

# Load datasets
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

# Preview data
print(train.head())
print(test.head())

# Check for missing values
print(train.isnull().sum())
print(test.isnull().sum())

# Basic statistics
print(train.describe())
```

## Model

- The app uses a Random Forest classifier.
- Preprocessing includes scaling numeric features and encoding categorical features.
- The trained model is serialized for deployment.

## How to Predict

- Open the Streamlit app.
- Enter feature values in the UI.
- Click "Predict Default" to see the result.

## License

This project is for educational purposes.
