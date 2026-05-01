import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt

print("1. Đọc dữ liệu và Chuẩn bị Features...")
df = pd.read_csv('data/processed/ML_READY_DATASET.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['code', 'date'])

# Bỏ các cột không phải là Feature (như tên nước, ngày hiện tại, và mục tiêu)
target = 'new_deaths_smoothed_per_million'
cols_to_drop = [target, 'code', 'country', 'date']
X = df.drop(columns=cols_to_drop)
y = df[target]

print(f"Tổng số Features sử dụng: {X.shape[1]}")

print("\n2. Chia tập Train/Test theo THỜI GIAN (80/20)...")
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Tập Train: {len(X_train)} ngày | Tập Test: {len(X_test)} ngày")

print("\n3. Đang huấn luyện [Mô hình Baseline] - Random Forest...")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)

print("\n4. Đang huấn luyện [Mô hình Sát thủ] - XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print("\n" + "="*40)
print("🏆 KẾT QUẢ SO SÁNH SAI SỐ")
print("="*40)
print(f"Random Forest  -> RMSE: {rf_rmse:.2f} | MAE: {rf_mae:.2f}")
print(f"XGBoost        -> RMSE: {xgb_rmse:.2f} | MAE: {xgb_mae:.2f}")

# Tính % cải thiện
improvement = ((rf_rmse - xgb_rmse) / rf_rmse) * 100
if improvement > 0:
    print(f"✅ XGBoost vượt trội hơn Random Forest {improvement:.1f}%")
else:
    print(f"⚠️ Random Forest đang làm tốt hơn XGBoost. Cần tinh chỉnh thêm XGBoost!")
print("="*40)

print("\n5. Đang chạy SHAP Explainer cho thuật toán thắng cuộc (XGBoost)...")
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# Vẽ biểu đồ mức độ quan trọng
plt.figure(figsize=(12, 10))

# Vẽ SHAP plot
# max_display=20: Chỉ hiện top 20 biến quan trọng nhất để tránh bị rối
shap.summary_plot(shap_values, X_test, show=False, max_display=20)

# Chỉnh sửa font chữ và tiêu đề
plt.title("Phân tích tầm quan trọng của các yếu tố (SHAP Summary Plot)", fontsize=16, pad=20)
plt.xlabel("SHAP Value (Tác động lên số ca tử vong)", fontsize=12)
plt.yticks(fontsize=10)

# Tự động căn chỉnh để không bị mất chữ ở rìa
plt.tight_layout()

# Lưu ảnh với độ phân giải cao (300 DPI là chuẩn in ấn khoa học)
plt.savefig('shap_high_res.png', dpi=300, bbox_inches='tight')
print("✅ Đã lưu ảnh chất lượng cao tại: 'shap_high_res.png'")
