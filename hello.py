import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings("ignore")

# Load Dataset
file_path = "test_data_cleaned.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Extract numerical values safely
def extract_numerical(value):
    try:
        return float(str(value).split()[0])
    except (ValueError, AttributeError, IndexError):
        return np.nan

numeric_columns = ["Mileage", "Engine", "Power", "New_Price"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].apply(extract_numerical)

# Drop rows with missing values in important numerical columns
df.dropna(subset=numeric_columns, inplace=True)

# One-Hot Encoding for categorical variables
categorical_columns = ["Name", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Save feature names for consistency
feature_names = list(df.columns)
feature_names.remove("New_Price")
joblib.dump(feature_names, "feature_names.pkl")

# Feature Scaling
scaler = StandardScaler()
scaled_features = ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
existing_scaled_features = [col for col in scaled_features if col in df.columns]
df[existing_scaled_features] = scaler.fit_transform(df[existing_scaled_features])

# Splitting the dataset
X = df[feature_names]
y = df["New_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42)
grid_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=5, cv=3, scoring='neg_mean_absolute_error', random_state=42)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Train XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_rf = best_rf.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"Random Forest - MAE: {mae_rf:.4f}, R2 Score: {r2_rf:.4f}")
print(f"XGBoost - MAE: {mae_xgb:.4f}, R2 Score: {r2_xgb:.4f}")

# Save models and scaler
joblib.dump(best_rf, "random_forest_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Models, scaler, and feature names saved successfully.")