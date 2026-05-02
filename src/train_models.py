import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt



# ===== DATA =====
print("Reading data and Preparing features...")
df = pd.read_csv('data/processed/ML_READY_DATASET.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['code', 'date'])

target = 'new_deaths_smoothed_per_million'
metadata = ['code', 'country', 'date']

# Present data leakage
leakage_cols = [
    # Policy
    'new_cases_smoothed_per_million', 'stringency_index',
    'C1M_School closing', 'C2M_Workplace closing', 'C3M_Cancel public events', 
    'C4M_Restrictions on gatherings', 'C5M_Close public transport', 
    'C6M_Stay at home requirements', 'C7M_Restrictions on internal movement', 
    'C8EV_International travel controls', 'H1_Public information campaigns', 
    'H2_Testing policy', 'H3_Contact tracing', 'H6M_Facial Coverings', 
    'H8M_Protection of elderly people', 'E1_Income support', 'E2_Debt/contract relief',

    # Mobility
    'retail_and_recreation_percent_change_from_baseline', 
    'grocery_and_pharmacy_percent_change_from_baseline', 
    'parks_percent_change_from_baseline', 
    'transit_stations_percent_change_from_baseline', 
    'workplaces_percent_change_from_baseline', 
    'residential_percent_change_from_baseline',

    # Tweets
    'daily_sentiment_avg', 'daily_sentiment_std', 'daily_tweet_volume', 
    'weighted_sentiment', 'sentiment_zscore', 
    'sentiment_delta_3', 'sentiment_delta_7', 'sentiment_roll7'
]

cols_to_drop = metadata + [target] + leakage_cols

X = df.drop(columns=cols_to_drop)
y = df[target]

print(f"Total number of Features: {X.shape[1]}")
# Check for leftovers leakage
print(X.columns.tolist())



# ===== TRAIN/TEST =====
print("\nTrain/Test Splitting...")
# Split by time (date, not rows): 80%
unique_dates = df['date'].sort_values().unique()
split_idx = int(len(unique_dates) * 0.8)
cutoff_date = unique_dates[split_idx]
print(f"Cut-off Date for Test: {pd.to_datetime(cutoff_date).date()}")

train_df = df[df['date'] < cutoff_date]
test_df = df[df['date'] >= cutoff_date]

X_train = train_df.drop(columns=cols_to_drop)
y_train = train_df[target]
X_test = test_df.drop(columns=cols_to_drop)
y_test = test_df[target]

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")



# ===== RANDOM FOREST =====
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)



# ===== XGBOOST =====
print("\nTraining XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,        # 
    learning_rate=0.03,      # 
    max_depth=4,             # 
    subsample=0.8,           # 
    colsample_bytree=0.8,    # 
    reg_alpha=0.5,           # L1 Regularization: Phạt mạnh các biến không quan trọng
    reg_lambda=1.0,          # L2 Regularization: Ngăn chặn trọng số quá lớn
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_mae = mean_absolute_error(y_test, xgb_preds)



# ===== EVALUATE =====
print("\n" + "="*40)
print("RESULTS")
print("="*40)
print(f"Random Forest  -> RMSE: {rf_rmse:.2f} | MAE: {rf_mae:.2f}")
print(f"XGBoost        -> RMSE: {xgb_rmse:.2f} | MAE: {xgb_mae:.2f}")

improvement = ((rf_rmse - xgb_rmse) / rf_rmse) * 100
if improvement > 0:
    print(f"XGBoost is better than Random Forest by {improvement:.1f}%")
else:
    print(f"Random Forest is doing better than XGBoost")
print("="*40)



# ===== SHAP =====
print("\nRunning SHAP...")
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

plt.figure(figsize=(12, 10))

# max_display=20
shap.summary_plot(shap_values, X_test, show=False, max_display=20)

plt.xlim(left=-2)

plt.title("Feature Importances (SHAP)", fontsize=16, pad=20)
plt.xlabel("SHAP Value (Impact on Deaths)", fontsize=12)
plt.yticks(fontsize=10)

plt.tight_layout()

plt.savefig('shap.png', dpi=300, bbox_inches='tight')
