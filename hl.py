import pandas as pd

# Đọc tệp dữ liệu
file_path = 'dataNCKH3.csv'  # Đảm bảo file nằm trong cùng thư mục làm việc
data = pd.read_csv(file_path, encoding='utf-8')

# Hiển thị thông tin cơ bản
data.info()

# Xem vài dòng đầu tiên
data.head()
# Kiểm tra kích thước dữ liệu
print(f"Kích thước dữ liệu: {data.shape}")

# Kiểm tra giá trị thiếu
print("Số lượng giá trị thiếu mỗi cột:")
print(data.isnull().sum())

# Hiển thị mô tả thống kê
data.describe()
from sklearn.model_selection import train_test_split

# Cột đầu vào và đầu ra
X = data[['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']]  # Thêm biến 'age'
y = data['strength']

# Phân chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Hiển thị kích thước tập dữ liệu
print(f"Kích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Khởi tạo mô hình LightGBM
model_lgb = lgb.LGBMRegressor(random_state=42)

# Huấn luyện mô hình
model_lgb.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_lgb = model_lgb.predict(X_test)

# Đánh giá mô hình
r2_lgb = r2_score(y_test, y_pred_lgb)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))

# Hiển thị kết quả
print(f"R^2 Score (LightGBM): {r2_lgb:.4f}")
print(f"Mean Absolute Error (MAE): {mae_lgb:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lgb:.4f}")
import joblib

# Lưu mô hình
joblib.dump(model_lgb, 'lightgbm_model.pkl')
print("Mô hình LightGBM đã được lưu thành công!")
