import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

# 1️⃣ Load dataset
df = pd.read_csv(r"E:\Project 497J\dataset1\dataset.csv", parse_dates=[
    "scheduled_departure_dt", "scheduled_arrival_dt", "date"
])

# 2️⃣ Define target and remove leakage columns
# Binary classification target: delayed if arrival_delay > 15
df["is_delayed"] = (df["arrival_delay"] > 15).astype(int)
df["delay_minutes"] = df["arrival_delay"].clip(lower=0)

leak_cols = [
    "departure_delay", "arrival_delay",
    "delay_carrier", "delay_weather",
    "delay_national_aviation_system",
    "delay_security", "delay_late_aircarft_arrival",
    "actual_departure_dt", "actual_arrival_dt"
]
df = df.drop(columns=leak_cols)

# Remove cancelled flights
df = df[df["cancelled_code"].isna()].drop(columns=["cancelled_code"])

# 3️⃣ Sort by time (so we can time-split later)
df = df.sort_values("scheduled_departure_dt")

# 4️⃣ Feature engineering – extract useful datetime parts
df["dep_hour"] = df["scheduled_departure_dt"].dt.hour
df["dep_minute"] = df["scheduled_departure_dt"].dt.minute
df["arr_hour"] = df["scheduled_arrival_dt"].dt.hour
df["arr_minute"] = df["scheduled_arrival_dt"].dt.minute

# 5️⃣ Select input features
cat_features = [
    "carrier_code", "flight_number", "origin_airport",
    "destination_airport", "tail_number", "weekday"
]
num_features = [
    "scheduled_elapsed_time", "year", "month", "day",
    "dep_hour", "arr_hour",
    "HourlyDryBulbTemperature_x", "HourlyPrecipitation_x",
    "HourlyStationPressure_x", "HourlyVisibility_x", "HourlyWindSpeed_x",
    "HourlyDryBulbTemperature_y", "HourlyPrecipitation_y",
    "HourlyStationPressure_y", "HourlyVisibility_y", "HourlyWindSpeed_y"
]

# 6️⃣ Split into train/test by time (e.g., last 20% = test)
split_point = int(len(df) * 0.8)
train_df = df.iloc[:split_point]
test_df = df.iloc[split_point:]

X_train = train_df[cat_features + num_features]
y_train_cls = train_df["is_delayed"]
y_train_reg = train_df["delay_minutes"]

X_test = test_df[cat_features + num_features]
y_test_cls = test_df["is_delayed"]
y_test_reg = test_df["delay_minutes"]

# 7️⃣ Preprocessing
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cat", cat_transformer, cat_features),
    ("num", num_transformer, num_features)
])

# 8️⃣ Classification model (is_delayed)
clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300, max_depth=6,
        learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    ))
])

clf.fit(X_train, y_train_cls)
y_pred_cls = clf.predict(X_test)
print("Classification Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print("F1 Score:", f1_score(y_test_cls, y_pred_cls))

# 9️⃣ Regression model (delay_minutes)
reg = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300, max_depth=6,
        learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    ))
])

reg.fit(X_train, y_train_reg)
y_pred_reg = reg.predict(X_test)
print("Regression MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
print("Regression RMSE:", mean_squared_error(y_test_reg, y_pred_reg, squared=False))
