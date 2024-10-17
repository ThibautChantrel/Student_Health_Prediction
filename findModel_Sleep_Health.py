import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the dataset
fileName = "Modified_sleep_health_and_lifestyle_dataset.csv"
FullPath = "./Modified_csv/" + fileName
df = pd.read_csv(FullPath, sep=';')  # Adjust the separator as needed

# Select the features and target variable (Stress Level)
X = df[['Gender', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Heart Rate', 'Daily Steps']]
y = df['Stress Level']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression': SVR(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor()
}

# Initialize a dictionary to store the results
model_performance = {}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_performance[model_name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

# Print the results in a more readable format
print("Model Comparison (sorted by RMSE):")
print("{:<25} {:<15} {:<15} {:<15} {:<10}".format('Model', 'MSE', 'RMSE', 'MAE', 'R²'))
print("="*80)
for model_name, metrics in sorted(model_performance.items(), key=lambda x: x[1]['RMSE']):
    print("{:<25} {:<15.4f} {:<15.4f} {:<15.4f} {:<10.4f}".format(
        model_name, metrics['MSE'], metrics['RMSE'], metrics['MAE'], metrics['R²']
    ))
