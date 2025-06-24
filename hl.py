import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# Đọc dữ liệu
file_path = 'dataNCKH3.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# Xác định X, y
X = data[['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']]
y = data['strength']

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Thiết lập lưới siêu tham số
param_grid = {
    'num_leaves': [15, 31, 50],
    'max_depth': [5, 10, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500]
}

# Khởi tạo LGBM và GridSearchCV
model = lgb.LGBMRegressor(random_state=42)
grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# Tìm tham số tối ưu
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print('Best parameters:', grid.best_params_)

# Đánh giá trên tập kiểm tra
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'R^2 Score: {r2:.4f}')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

# Cross-validation với mô hình tốt nhất trên toàn bộ data
cv_rmse = -cross_val_score(best_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f'CV RMSE (mean ± std): {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}')

# Lưu mô hình
joblib.dump(best_model, 'lgbm_best_model.pkl')
print('Mô hình LightGBM tối ưu đã được lưu!')
from sklearn.metrics import mean_squared_error

# Đánh giá trên tập kiểm tra
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f'R² (R2 Score): {r2:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
