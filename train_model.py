import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def load_data_from_folder(folder_path):
    combined_data = pd.DataFrame()

    # Recursively traverse all subdirectories and load .txt files
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                
                # Load each .txt file
                df = pd.read_csv(file_path)

                # Clean up column names
                df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
                if 'date' in df.columns.str.lower():
                    df.rename(columns={col: 'Date' for col in df.columns if col.lower() == 'date'}, inplace=True)

                # Add a column for the asset name based on the filename
                df['Asset'] = filename.split('.')[0]
                
                # Append to the combined DataFrame
                combined_data = pd.concat([combined_data, df], ignore_index=True)
    
    return combined_data

# Load data from the main stock_data folder
data_folder_path = 'data/stock_data'  # Adjust this path if necessary
stock_data = load_data_from_folder(data_folder_path)

# Convert 'Date' column to datetime and sort by date
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data = stock_data.sort_values(by=['Asset', 'Date'])

# Drop 'OpenInt' if not needed
stock_data.drop(columns=['OpenInt'], inplace=True, errors='ignore')

# Feature Engineering: Add moving averages, daily return, and volatility
stock_data['MA10'] = stock_data.groupby('Asset')['Close'].transform(lambda x: x.rolling(window=10).mean())
stock_data['MA50'] = stock_data.groupby('Asset')['Close'].transform(lambda x: x.rolling(window=50).mean())
stock_data['Daily Return'] = stock_data.groupby('Asset')['Close'].transform(lambda x: x.pct_change())
stock_data['Volatility'] = stock_data.groupby('Asset')['Close'].transform(lambda x: x.rolling(window=10).std())

# Drop rows with NaN values created by rolling calculations
stock_data.dropna(inplace=True)

# Filter data for a specific asset (e.g., 'AAPL') for training
asset_data = stock_data[stock_data['Asset'] == 'AAPL']  # Replace 'AAPL' with any specific stock if needed
X = asset_data[['MA10', 'MA50', 'Daily Return', 'Volatility']]
y = asset_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# Save the model and scaler
joblib.dump(model, 'models/stock_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
