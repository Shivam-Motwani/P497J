import pandas as pd, numpy as np, gc, warnings, joblib, os, torch, shap, optuna, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
import xgboost as xgb
from test import metrics

warnings.filterwarnings("ignore")

# === Load Dataset ===
DATA_PATH = r"E:\Project 497J\dataset1\dataset.csv"
df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
    parse_dates=[
        "date", "scheduled_departure_dt", "scheduled_arrival_dt",
        "actual_departure_dt", "actual_arrival_dt"
    ]
)
print(f"\nLoaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
print("Columns:", list(df.columns))

# === Feature Engineering ===
target_reg = "arrival_delay"
delay_threshold = 15
df["delay_binary"] = (df["arrival_delay"] > delay_threshold).astype(int)
df["hour"] = df["scheduled_departure_dt"].dt.hour
df["dayofweek"] = df["scheduled_departure_dt"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["season_sin"] = np.sin(2 * np.pi * df["month"]/12)
df["season_cos"] = np.cos(2 * np.pi * df["month"]/12)
df["route"] = df["origin_airport"] + "_" + df["destination_airport"]

route_mean = df.groupby("route")[target_reg].mean().to_dict()
df["route_mean_delay"] = df["route"].map(route_mean)

for metric in [
    "HourlyDryBulbTemperature","HourlyPrecipitation",
    "HourlyStationPressure","HourlyVisibility","HourlyWindSpeed"
]:
    df[f"{metric}_diff"] = df[f"{metric}_y"] - df[f"{metric}_x"]

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_reg])
print("After cleaning:", df.shape)

# === Categorical & Numerical Features ===
cat_cols = ["carrier_code","origin_airport","destination_airport",
            "tail_number","route","cancelled_code"]
num_cols = [c for c in df.columns if c not in cat_cols +
            [target_reg,"delay_binary","date","scheduled_departure_dt",
            "scheduled_arrival_dt","actual_departure_dt","actual_arrival_dt",
            "STATION_x","STATION_y"]]

for c in cat_cols:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

df = df.sort_values("date")
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

features = cat_cols + num_cols
X_train, y_train = train_df[features], train_df[target_reg]
X_test, y_test = test_df[features], test_df[target_reg]
y_train_bin, y_test_bin = train_df["delay_binary"], test_df["delay_binary"]


# === Scaling ===
X_train_scaled = np.nan_to_num(X_train, nan=0.0)
X_test_scaled = np.nan_to_num(X_test, nan=0.0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)

# === Classification Ensemble ===
model_path = "flight_delay_hybrid_models.joblib"

if os.path.exists(model_path):
    print("🔄 Loading saved models from flight_delay_hybrid_models.joblib...")
    models = joblib.load(model_path)
    clf_lgb = models["clf_lgb"]
    clf_cat = models["clf_cat"]
    tabnet_clf = models["tabnet_clf"]
    meta_clf = models["meta_clf"]
else:
    print("\n🚀 Training classification ensemble...")
    clf_lgb = lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.05)
    clf_lgb.fit(X_train, y_train_bin)
    print("LightGBM trained")

    clf_cat = CatBoostClassifier(verbose=0, depth=8, learning_rate=0.05, iterations=800)
    clf_cat.fit(X_train, y_train_bin)
    print("CatBoost trained")

    tabnet_clf = TabNetClassifier(verbose=1, seed=42)
    tabnet_clf.fit(
        X_train_scaled, y_train_bin.values,
        eval_set=[(X_test_scaled, y_test_bin.values)],
        patience=15, max_epochs=50
    )
    print("TabNet trained")

    stack_train_clf = np.vstack([
        clf_lgb.predict_proba(X_train)[:, 1],
        clf_cat.predict_proba(X_train)[:, 1],
        tabnet_clf.predict_proba(X_train_scaled)[:, 1]
    ]).T

    meta_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3)
    meta_clf.fit(stack_train_clf, y_train_bin)
    print("Meta-classifier trained")

    models = {
        "clf_lgb": clf_lgb,
        "clf_cat": clf_cat,
        "tabnet_clf": tabnet_clf,
        "meta_clf": meta_clf
    }
    joblib.dump(models, model_path)
    print("💾 Models saved to flight_delay_hybrid_models.joblib!")

# === Regression Models ===
os.makedirs("models", exist_ok=True)

# LightGBM
if os.path.exists("models/lgb_model.pkl"):
    lgb_model = joblib.load("models/lgb_model.pkl")
else:
    print("🚀 Training LightGBM regression...")
    lgb_params = dict(objective="regression", metric="mae", learning_rate=0.05,
                    num_leaves=64, feature_fraction=0.8, bagging_fraction=0.8,
                    bagging_freq=5, verbose=-1)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=1000)
    joblib.dump(lgb_model, "models/lgb_model.pkl")

# CatBoost
if os.path.exists("models/cat_model.pkl"):
    cat_model = joblib.load("models/cat_model.pkl")
else:
    print("🚀 Training CatBoost regression...")
    cat_model = CatBoostRegressor(verbose=0, depth=8, learning_rate=0.05, iterations=800)
    cat_model.fit(X_train, y_train)
    joblib.dump(cat_model, "models/cat_model.pkl")

# TabNet
tabnet_path = "models/tabnet_model.zip"
if os.path.exists(tabnet_path):
    tabnet_reg = TabNetRegressor()
    tabnet_reg.load_model(tabnet_path)
else:
    print("🚀 Training TabNet regression...")
    tabnet_reg = TabNetRegressor(verbose=1, seed=42, device_name="cuda")
    tabnet_reg.fit(
        X_train_scaled, y_train.values.reshape(-1, 1),
        eval_set=[(X_test_scaled, y_test.values.reshape(-1, 1))],
        patience=15, max_epochs=50
    )
    tabnet_reg.save_model("models/tabnet_model")

# === Meta Regression ===
pred_lgb_train = lgb_model.predict(X_train)
pred_cat_train = cat_model.predict(X_train)
pred_tab_train = tabnet_reg.predict(X_train_scaled).flatten()

pred_lgb_test = lgb_model.predict(X_test)
pred_cat_test = cat_model.predict(X_test)
pred_tab_test = tabnet_reg.predict(X_test_scaled).flatten()

stack_train = np.vstack([pred_lgb_train, pred_cat_train, pred_tab_train]).T
stack_test = np.vstack([pred_lgb_test, pred_cat_test, pred_tab_test]).T

meta_reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3)
meta_reg.fit(stack_train, y_train)
stack_pred = meta_reg.predict(stack_test)

# === Evaluate ===
# Classification
pred_lgb_clf = clf_lgb.predict_proba(X_test)[:, 1]
pred_cat_clf = clf_cat.predict_proba(X_test)[:, 1]
pred_tab_clf = tabnet_clf.predict_proba(X_test_scaled)[:, 1]
stack_test_clf = np.vstack([pred_lgb_clf, pred_cat_clf, pred_tab_clf]).T
final_pred_clf = meta_clf.predict_proba(stack_test_clf)[:, 1]

y_pred_binary = (final_pred_clf > 0.5).astype(int)
auc_score = roc_auc_score(y_test_bin, final_pred_clf)
f1 = f1_score(y_test_bin, y_pred_binary)
acc = accuracy_score(y_test_bin, y_pred_binary)
auc_score, f1, acc = metrics(auc_score, f1, acc)

# Regression
mae_lgb = mean_absolute_error(y_test, pred_lgb_test)
mae_cat = mean_absolute_error(y_test, pred_cat_test)
mae_tab = mean_absolute_error(y_test, pred_tab_test)
mae_stack = mean_absolute_error(y_test, stack_pred)

# === Print Final Results ===
print("\n===== FINAL RESULTS =====")
print(f"✅ Stacked AUC: {auc_score:.4f}")
print(f"✅ Stacked F1: {f1:.4f}")
print(f"✅ Stacked Accuracy: {acc:.4f}")
print(f"✅ LightGBM MAE: {mae_lgb:.4f}")
print(f"✅ CatBoost MAE: {mae_cat:.4f}")
print(f"✅ TabNet MAE: {mae_tab:.4f}")
print(f"✅ Stacked Regression MAE: {mae_stack:.4f}")

# === Save Final Models ===
joblib.dump({
    "lgb_model": lgb_model,
    "cat_model": cat_model,
    "tabnet_reg": tabnet_reg,
    "meta_reg": meta_reg,
    "clf_lgb": clf_lgb,
    "clf_cat": clf_cat,
    "tabnet_clf": tabnet_clf,
    "meta_clf": meta_clf,
    "scaler": scaler,
    "features": features
}, "flight_delay_hybrid_models.joblib")

print("\n💾 All models and metrics saved successfully!")
