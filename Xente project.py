#%%
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib_inline as mplib
# Define the path to the train dataset
train_path = r"C:\Users\user\OneDrive\Desktop\DS\Project\Train.csv"
# Load the train dataset
train = pd.read_csv(train_path)
# Display the first few rows in the Streamlit app and in the terminal
st.write("Preview of the train dataset:")
st.write(train.head())
print("Preview of the train dataset:")
print(train.head())
# Data Cleaning for the train dataset

# 1. Remove duplicate rows
train = train.drop_duplicates()

# 2. Handle missing values
# For numeric columns, fill with median
num_cols = train.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    train[col] = train[col].fillna(train[col].median())

# For categorical columns, fill with mode
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])

# 3. Remove columns with too many missing values (optional, e.g., >50%)
missing_thresh = 0.5
train = train.loc[:, train.isnull().mean() < missing_thresh]

# Truncate ProductId to remove the string and keep only the numeric part
# Example: 'ProductId_123' -> 123

if 'ProductId' in train.columns:
    train['ProductId_numeric'] = train['ProductId'].str.extract('(\d+)', expand=False).astype(float)

# Display result
st.write("ProductId truncated to numeric:")
st.write(train[['ProductId', 'ProductId_numeric']].head())
print("ProductId truncated to numeric:")
print(train[['ProductId', 'ProductId_numeric']].head())
# Replace ProductCategory with numerics using a mapping
category_map = {
    'airtime': 1,
    'data bundles': 2,
    'financial services': 3,
    'movies': 4,
    'retail': 5,
    'tv': 6,
    'utility bill': 7
}

if 'ProductCategory' in train.columns:
    train['ProductCategory_numeric'] = train['ProductCategory'].map(category_map)

    st.write("ProductCategory replaced with numerics:")
    st.write(train[['ProductCategory', 'ProductCategory_numeric']].head())
    print("ProductCategory replaced with numerics:")
    print(train[['ProductCategory', 'ProductCategory_numeric']].head())
else:
    st.write("ProductCategory column not found in the dataset.")
    print("ProductCategory column not found in the dataset.")
# Display cleaned data
st.write("Cleaned train dataset preview:")
st.write(train.head())
print("Cleaned train dataset preview:")
print(train.head())
# EDA for the train dataset
# 1. Distribution of defulted loans
# Bar graph for distribution of defaulted loans

if 'IsDefaulted' in train.columns:
    st.write("Bar Graph: Distribution of Defaulted Loans")
    fig, ax = plt.subplots()
    train['IsDefaulted'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel('IsDefaulted')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Defaulted Loans')
    st.pyplot(fig)

    print("Distribution of Defaulted Loans:")
    print(train['IsDefaulted'].value_counts())
else:
    st.write("IsDefaulted column not found in the dataset.")
    print("IsDefaulted column not found")
# 2. Distribution of ProductCategory
# Pie chart for distribution of ProductCategory
if 'ProductCategory' in train.columns:
    st.write("Pie Chart: Distribution of ProductCategory")
    fig, ax = plt.subplots()
    train['ProductCategory'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
    ax.set_ylabel('')
    ax.set_title('Distribution of ProductCategory')
    st.pyplot(fig)

    print("Distribution of ProductCategory:")
    print(train['ProductCategory'].value_counts())
else:
    st.write("ProductCategory column not found in the dataset.")
    print("ProductCategory column not found in the dataset.")
# 3. ProductId distribution
# Histogram for ProductId distribution
# Bar Plot: Distribution of ProductId (using string labels)

import seaborn as sns

if 'ProductId' in train.columns:
    st.write("Bar Plot: Distribution of ProductId (String Labels)")
    # Count occurrences of each ProductId
    product_counts = train['ProductId'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    # Use a color palette for the bars
    colors = sns.color_palette("husl", len(product_counts))
    sns.barplot(x=product_counts.index, y=product_counts.values, palette=colors, ax=ax)
    ax.set_xlabel('ProductId')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of ProductId')
    # Optionally, limit the number of x-ticks for readability
    if len(product_counts) > 20:
        ax.set_xticks(ax.get_xticks()[::max(1, len(product_counts)//20)])
        ax.set_xticklabels([product_counts.index[int(i)] for i in ax.get_xticks()], rotation=45, ha='right')
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

    print("Distribution of ProductId (string):")
    print(product_counts)
else:
    st.write("ProductId column not found in the dataset.")
    print("ProductId column not found")
# 4. Average loan value
# Calculate and display the average loan value from the Value column
if 'Value' in train.columns:
    average_loan_value = train['Value'].mean()
    st.write(f"Average Loan Value: {average_loan_value}")
    print(f"Average Loan Value: {average_loan_value}")
else:
    st.write("Value column not found in the dataset.")
    print("Value column not found in the dataset.")
# 5. Loan value distribution
# 5. Loan value distribution as bar graphs (with different colors for each interval)

if 'Value' in train.columns:
    # Define intervals (bins)
    bins = [0, 25000, 50000, 100000, 200000, train['Value'].max()]
    labels = ['0-25k', '25k-50k', '50k-100k', '100k-200k', '200k+']
    train['LoanValueInterval'] = pd.cut(train['Value'], bins=bins, labels=labels, include_lowest=True)

    st.write("Loan Value Interval Distribution (Bar Graph):")
    interval_counts = train['LoanValueInterval'].value_counts().reindex(labels)
    fig, ax = plt.subplots()
    # Use a color palette for the bars
    colors = sns.color_palette("husl", len(interval_counts))
    bars = ax.bar(interval_counts.index, interval_counts.values, color=colors)
    ax.set_xlabel('Loan Value Interval')
    ax.set_ylabel('Count')
    ax.set_title('Loan Value Distribution by Interval')
    st.pyplot(fig)

    print("Loan Value Interval Distribution:")
    print(interval_counts)
else:
    st.write("Value column not found in the dataset.")
    print("Value column not found in the dataset.")
# Repayment time classification: early, on time, or late
## Repayment time classification: early, on time, or late using PaidOnDate and DueDate

if 'PaidOnDate' in train.columns and 'DueDate' in train.columns:
    # Convert to datetime if not already
    train['PaidOnDate'] = pd.to_datetime(train['PaidOnDate'])
    train['DueDate'] = pd.to_datetime(train['DueDate'])

    # Classify repayment status, including "On Time" for loans paid exactly on the due date
    def classify_repayment(row):
        if pd.isnull(row['PaidOnDate']) or pd.isnull(row['DueDate']):
            return 'Unknown'
        elif row['PaidOnDate'] < row['DueDate']:
            return 'Early'
        elif row['PaidOnDate'] == row['DueDate']:
            return 'On Time'
        else:
            return 'Late'

    train['RepaymentStatus'] = train.apply(classify_repayment, axis=1)

    st.write("Repayment Status Distribution (using PaidOnDate and DueDate):")
    st.write(train['RepaymentStatus'].value_counts())
    print("Repayment Status Distribution (using PaidOnDate and DueDate):")
    print(train['RepaymentStatus'].value_counts())

    # Bar plot for repayment status
    fig, ax = plt.subplots()
    colors = sns.color_palette("husl", len(train['RepaymentStatus'].unique()))
    train['RepaymentStatus'].value_counts().plot(kind='bar', color=colors, ax=ax)
    ax.set_xlabel('Repayment Status')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Repayment Status')
    st.pyplot(fig)
else:
    st.write("PaidOnDate or DueDate column not found in the dataset.")
    print("PaidOnDate or DueDate column not found in the dataset.")
# Histogram for days late or early

if 'PaidOnDate' in train.columns and 'DueDate' in train.columns:
    # Convert to datetime if not already
    train['PaidOnDate'] = pd.to_datetime(train['PaidOnDate'])
    train['DueDate'] = pd.to_datetime(train['DueDate'])

    # Calculate days late or early
    train['Days_Late'] = (train['PaidOnDate'] - train['DueDate']).dt.days

    st.write("Histogram: Distribution of Loan Repayment Days (Late or Early)")
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(train['Days_Late'], bins=50, kde=True, ax=ax)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title('Distribution of Loan Repayment Days (Late or Early)')
    ax.set_xlabel('Days Late (negative = early)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    print("Histogram for Days_Late plotted.")
else:
    st.write("PaidOnDate or DueDate column not found in the dataset.")
    print("PaidOnDate or DueDate column not found in the dataset.")
# Correlation analysis
# Correlation analysis between categorical/numeric columns and IsDefaulted

# ProductCategory vs IsDefaulted (using numeric encoding if available)
if 'ProductCategory_numeric' in train.columns and 'IsDefaulted' in train.columns:
    corr_pc = train['ProductCategory_numeric'].corr(train['IsDefaulted'])
    st.write(f"Correlation between ProductCategory (numeric) and IsDefaulted: {corr_pc:.3f}")
    print(f"Correlation between ProductCategory (numeric) and IsDefaulted: {corr_pc:.3f}")
else:
    st.write("ProductCategory_numeric or IsDefaulted column not found for correlation.")
    print("ProductCategory_numeric or IsDefaulted column not found for correlation.")

# ProductId vs IsDefaulted (using numeric encoding if available)
if 'ProductId_numeric' in train.columns and 'IsDefaulted' in train.columns:
    corr_pid = train['ProductId_numeric'].corr(train['IsDefaulted'])
    st.write(f"Correlation between ProductId (numeric) and IsDefaulted: {corr_pid:.3f}")
    print(f"Correlation between ProductId (numeric) and IsDefaulted: {corr_pid:.3f}")
else:
    st.write("ProductId_numeric or IsDefaulted column not found for correlation.")
    print("ProductId_numeric or IsDefaulted column not found for correlation.")

# Value vs IsDefaulted
if 'Value' in train.columns and 'IsDefaulted' in train.columns:
    corr_val = train['Value'].corr(train['IsDefaulted'])
    st.write(f"Correlation between Value and IsDefaulted: {corr_val:.3f}")
    print(f"Correlation between Value and IsDefaulted: {corr_val:.3f}")
else:
    st.write("Value or IsDefaulted column not found for correlation.")
    print("Value or IsDefaulted column not found for correlation.")

# BatchId vs IsDefaulted (using numeric encoding if needed)
if 'BatchId' in train.columns and 'IsDefaulted' in train.columns:
    # Convert BatchId to numeric if not already
    if not pd.api.types.is_numeric_dtype(train['BatchId']):
        train['BatchId_numeric'] = train['BatchId'].astype('category').cat.codes
        corr_batch = train['BatchId_numeric'].corr(train['IsDefaulted'])
        st.write(f"Correlation between BatchId (numeric) and IsDefaulted: {corr_batch:.3f}")
        print(f"Correlation between BatchId (numeric) and IsDefaulted: {corr_batch:.3f}")
    else:
        corr_batch = train['BatchId'].corr(train['IsDefaulted'])
        st.write(f"Correlation between BatchId and IsDefaulted: {corr_batch:.3f}")
        print(f"Correlation between BatchId and IsDefaulted: {corr_batch:.3f}")
else:
    st.write("BatchId or IsDefaulted column not found for correlation.")
    print("BatchId or IsDefaulted column not found for correlation.")
# ThirdPartyId vs IsDefaulted (using numeric encoding if needed)
if 'ThirdPartyId' in train.columns and 'IsDefaulted' in train.columns:
    if not pd.api.types.is_numeric_dtype(train['ThirdPartyId']):
        train['ThirdPartyId_numeric'] = train['ThirdPartyId'].astype('category').cat.codes
        corr_thirdparty = train['ThirdPartyId_numeric'].corr(train['IsDefaulted'])
        st.write(f"Correlation between ThirdPartyId (numeric) and IsDefaulted: {corr_thirdparty:.3f}")
        print(f"Correlation between ThirdPartyId (numeric) and IsDefaulted: {corr_thirdparty:.3f}")
    else:
        corr_thirdparty = train['ThirdPartyId'].corr(train['IsDefaulted'])
        st.write(f"Correlation between ThirdPartyId and IsDefaulted: {corr_thirdparty:.3f}")
        print(f"Correlation between ThirdPartyId and IsDefaulted: {corr_thirdparty:.3f}")
else:
    st.write("ThirdPartyId or IsDefaulted column not found for correlation.")
    print("ThirdPartyId or IsDefaulted column not found for correlation.")

# InvestorId vs IsDefaulted (using numeric encoding if needed)
if 'InvestorId' in train.columns and 'IsDefaulted' in train.columns:
    if not pd.api.types.is_numeric_dtype(train['InvestorId']):
        train['InvestorId_numeric'] = train['InvestorId'].astype('category').cat.codes
        corr_investor = train['InvestorId_numeric'].corr(train['IsDefaulted'])
        st.write(f"Correlation between InvestorId (numeric) and IsDefaulted: {corr_investor:.3f}")
        print(f"Correlation between InvestorId (numeric) and IsDefaulted: {corr_investor:.3f}")
    else:
        corr_investor = train['InvestorId'].corr(train['IsDefaulted'])
        st.write(f"Correlation between InvestorId and IsDefaulted: {corr_investor:.3f}")
        print(f"Correlation between InvestorId and IsDefaulted: {corr_investor:.3f}")
else:
    st.write("InvestorId or IsDefaulted column not found for correlation.")
    print("InvestorId or IsDefaulted column not found for correlation.")

# SubscriptionId vs IsDefaulted (using numeric encoding if needed)
if 'SubscriptionId' in train.columns and 'IsDefaulted' in train.columns:
    if not pd.api.types.is_numeric_dtype(train['SubscriptionId']):
        train['SubscriptionId_numeric'] = train['SubscriptionId'].astype('category').cat.codes
        corr_sub = train['SubscriptionId_numeric'].corr(train['IsDefaulted'])
        st.write(f"Correlation between SubscriptionId (numeric) and IsDefaulted: {corr_sub:.3f}")
        print(f"Correlation between SubscriptionId (numeric) and IsDefaulted: {corr_sub:.3f}")
    else:
        corr_sub = train['SubscriptionId'].corr(train['IsDefaulted'])
        st.write(f"Correlation between SubscriptionId and IsDefaulted: {corr_sub:.3f}")
        print(f"Correlation between SubscriptionId and IsDefaulted: {corr_sub:.3f}")
else:
    st.write("SubscriptionId or IsDefaulted column not found for correlation.")
    print("SubscriptionId or IsDefaulted column not found for correlation.")
# Correlation heatmap for relevant columns

# Select columns for correlation analysis
corr_columns = [
    'IsDefaulted',
    'ProductCategory_numeric',
    'ProductId_numeric',
    'Value',
    'BatchId_numeric' if 'BatchId_numeric' in train.columns else 'BatchId',
    'ThirdPartyId_numeric' if 'ThirdPartyId_numeric' in train.columns else 'ThirdPartyId',
    'InvestorId_numeric' if 'InvestorId_numeric' in train.columns else 'InvestorId',
    'SubscriptionId_numeric' if 'SubscriptionId_numeric' in train.columns else 'SubscriptionId'
]

# Filter only columns that exist in the DataFrame
corr_columns = [col for col in corr_columns if col in train.columns]

if len(corr_columns) > 1:
    corr_matrix = train[corr_columns].corr()
    st.write("Correlation Heatmap:")
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    print("Correlation heatmap displayed.")
else:
    st.write("Not enough columns for correlation heatmap.")
    print("Not enough columns for correlation heatmap.")
# Feature Creation using columns in the train dataset

# 1. Loan Duration (in days)
if 'DisbursementDate' in train.columns and 'DueDate' in train.columns:
    train['DisbursementDate'] = pd.to_datetime(train['DisbursementDate'])
    train['DueDate'] = pd.to_datetime(train['DueDate'])
    train['LoanDuration'] = (train['DueDate'] - train['DisbursementDate']).dt.days
    st.write("LoanDuration feature created (days):")
    st.write(train[['DisbursementDate', 'DueDate', 'LoanDuration']].head())
    print("LoanDuration feature created (days):")
    print(train[['DisbursementDate', 'DueDate', 'LoanDuration']].head())

# 2. Days Late (already created above, but ensure it's present)
if 'PaidOnDate' in train.columns and 'DueDate' in train.columns:
    train['PaidOnDate'] = pd.to_datetime(train['PaidOnDate'])
    train['DueDate'] = pd.to_datetime(train['DueDate'])
    train['Days_Late'] = (train['PaidOnDate'] - train['DueDate']).dt.days
    st.write("Days_Late feature created:")
    st.write(train[['PaidOnDate', 'DueDate', 'Days_Late']].head())
    print("Days_Late feature created:")
    print(train[['PaidOnDate', 'DueDate', 'Days_Late']].head())

# 3. IsWeekendDisbursement (1 if disbursed on weekend, else 0)
if 'DisbursementDate' in train.columns:
    train['IsWeekendDisbursement'] = pd.to_datetime(train['DisbursementDate']).dt.weekday >= 5
    train['IsWeekendDisbursement'] = train['IsWeekendDisbursement'].astype(int)
    st.write("IsWeekendDisbursement feature created:")
    st.write(train[['DisbursementDate', 'IsWeekendDisbursement']].head())
    print("IsWeekendDisbursement feature created:")
    print(train[['DisbursementDate', 'IsWeekendDisbursement']].head())

# 4. LoanToValueRatio (if 'Value' and 'LoanAmount' columns exist)
if 'Value' in train.columns and 'LoanAmount' in train.columns:
    train['LoanToValueRatio'] = train['Value'] / train['LoanAmount']
    st.write("LoanToValueRatio feature created:")
    st.write(train[['Value', 'LoanAmount', 'LoanToValueRatio']].head())
    print("LoanToValueRatio feature created:")
    print(train[['Value', 'LoanAmount', 'LoanToValueRatio']].head())

# 5. Number of Previous Loans by Customer (if 'CustomerId' exists)
if 'CustomerId' in train.columns:
    train['NumPrevLoans'] = train.groupby('CustomerId').cumcount()
    st.write("NumPrevLoans feature created:")
    st.write(train[['CustomerId', 'NumPrevLoans']].head())
# 6. Check if a customer borrowed for different products using CustomerId and ProductId

if 'CustomerId' in train.columns and 'ProductId' in train.columns:
    # Count the number of unique products each customer has borrowed
    customer_product_counts = train.groupby('CustomerId')['ProductId'].nunique()
    # Create a flag: 1 if customer borrowed more than one product, else 0
    train['BorrowedMultipleProducts'] = train['CustomerId'].map(lambda x: 1 if customer_product_counts[x] > 1 else 0)

    st.write("Sample of BorrowedMultipleProducts feature:")
    st.write(train[['CustomerId', 'ProductId', 'BorrowedMultipleProducts']].head(10))
    print("Sample of BorrowedMultipleProducts feature:")
    print(train[['CustomerId', 'ProductId', 'BorrowedMultipleProducts']].head(10))
else:
    st.write("CustomerId or ProductId column not found in the dataset.")
    print("CustomerId or ProductId column not found in the dataset.")
#%%
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# Define the path to the train dataset
train_path = r"C:\Users\user\OneDrive\Desktop\DS\Project\Train.csv"
train = pd.read_csv(train_path)

# Variables to eliminate
eliminate_vars = [
    'CustomerId', 'TransactionStartTime', 'Value', 'TransactionId', 'CurrencyCode', 'CountryCode',
    'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'TransactionStatus', 'IssuedDateLoan',
    'AmountLoan', 'Currency', 'LoanId', 'PaidOnDate', 'IsFinalPayBack', 'LoanApplicationId', 'PayBackId'
]

# Remove unwanted columns
train = train.drop(columns=[col for col in eliminate_vars if col in train.columns])

# Data Cleaning
train = train.drop_duplicates()
num_cols = train.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    train[col] = train[col].fillna(train[col].median())
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
missing_thresh = 0.5
train = train.loc[:, train.isnull().mean() < missing_thresh]

# Label Encoding for categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le

# Standard Scaling for numeric columns (excluding target and already encoded columns)
exclude_cols = ['IsDefaulted', 'Good_Bad_flag']
num_cols = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col not in exclude_cols]
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])

# Feature Selection
target_col = 'IsDefaulted'
feature_cols = [col for col in train.columns if col != target_col and train[col].dtype in [int, float]]
train_fs = train.dropna(subset=feature_cols + [target_col])
X = train_fs[feature_cols]
y = train_fs[target_col]

# Feature Importance (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Split train dataset: 70% training, 15% validation, 15% testing
train_set, temp_set = train_test_split(
    train, test_size=0.3, random_state=42, stratify=train['IsDefaulted'] if 'IsDefaulted' in train.columns else None
)
val_set, test_set = train_test_split(
    temp_set, test_size=0.5, random_state=42, stratify=temp_set['IsDefaulted'] if 'IsDefaulted' in temp_set.columns else None
)

# Model training using Random Forest
features = [col for col in train_set.columns if col not in ['IsDefaulted', 'Good_Bad_flag'] and train_set[col].dtype in [int, float]]
X_train = train_set[features]
y_train = train_set['IsDefaulted']
X_val = val_set[features]
y_val = val_set['IsDefaulted']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_val)
y_proba = rf_model.predict_proba(X_val)[:, 1]

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, 
    n_iter=10, scoring='roc_auc', 
    cv=3, random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_
y_pred_best = best_rf.predict(X_val)
y_proba_best = best_rf.predict_proba(X_val)[:, 1]

# Model evaluation using the unseen test data
test_features = [col for col in test_set.columns if col in X_train.columns]
X_test = test_set[test_features]
y_test = test_set['IsDefaulted']
y_pred_test = best_rf.predict(X_test)
y_proba_test = best_rf.predict_proba(X_test)[:, 1]

import pickle

# Save the best Random Forest model to a file
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
st.write("Model serialized and saved as best_rf_model.pkl")
print("Model serialized and saved as best_rf_model.pkl")

# Streamlit App
st.title("Xente Credit Default Prediction (Cleaned Features)")

user_input = {}
orig_train = pd.read_csv(train_path)  # Load original data once

for feat in features:
    if feat in cat_cols:
        options = orig_train[feat].dropna().unique().tolist()
        user_input[feat] = st.selectbox(f"{feat}", options=options)
    elif feat in ['BatchId', 'InvestorId', 'ThirdPartyId', 'SubscriptionId']:
        options = orig_train[feat].dropna().unique().tolist()
        user_input[feat] = st.selectbox(f"{feat}", options=options)
    elif feat == 'DueDate':
        min_date = pd.to_datetime(orig_train[feat]).min()
        max_date = pd.to_datetime(orig_train[feat]).max()
        user_input[feat] = st.date_input(f"{feat}", value=min_date, min_value=min_date, max_value=max_date)
    else:
        min_val = int(orig_train[feat].min())
        max_val = int(orig_train[feat].max())
        mean_val = int(orig_train[feat].mean())
        user_input[feat] = st.slider(f"{feat}", min_value=min_val, max_value=max_val, value=mean_val, step=1)
if st.button("Predict Default"):
    input_df = pd.DataFrame([user_input])
    # Label encoding for categorical columns
    for col in cat_cols:
        if col in input_df.columns:
            le = label_encoders[col]
            # If value not in classes, assign -1 or most frequent class
            input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    # Standard scaling for numeric columns
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    # Ensure all features used during fit are present in input_df
    for col in features:
        if col not in input_df.columns:
            if col in num_cols:
                input_df[col] = 0
            else:
                input_df[col] = ''
    input_df = input_df[features]
    pred = best_rf.predict(input_df)[0]
    prob = best_rf.predict_proba(input_df)[0][1]
    if pred == 1:
        st.success(f"Prediction: Defaulted (Probability: {prob:.2f})")
    else:
        st.info(f"Prediction: Not Defaulted (Probability: {prob:.2f})")

# Display feature importance
st.write("Top Features by Model-Based Importance:")
st.write(feature_importance_df.head(10))

# Display model evaluation
st.write("Random Forest Results (Validation Set):")
st.write(classification_report(y_val, y_pred_best))
st.write(f"Validation ROC-AUC: {roc_auc_score(y_val, y_proba_best):.3f}")
st.write("Test Results with Best Random Forest:")
st.write(classification_report(y_test, y_pred_test))
st.write(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba_test):.3f}")