import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. 读取数据
df = pd.read_csv("data/raw/hour.csv")
df['dteday'] = pd.to_datetime(df['dteday'])

# 2. 特征选择（用天气、时间等特征来预测租车数 cnt）
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
target = 'cnt'

X = df[features]
y = df[target]

# 3. 时间正确划分（极其重要！千万不能用随机划分）
# 用前 2 年的数据训练，最后 4 个月的数据测试
split_date = '2012-09-01'
train_mask = df['dteday'] < split_date
test_mask = df['dteday'] >= split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# 4. 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 在测试集上做预测
y_pred = model.predict(X_test)

# 6. 计算 MAE 和 RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"训练集大小: {len(X_train)} 行")
print(f"测试集大小: {len(X_test)} 行")
print(f"线性回归基线模型结果：")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

from sklearn.ensemble import RandomForestRegressor

print("\n--- 开始训练随机森林模型 ---")
# 随机森林模型：100棵树，固定随机种子保证结果可复现
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 预测与评估
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"随机森林对比结果：")
print(f"MAE: {rf_mae:.2f}")
print(f"RMSE: {rf_rmse:.2f}")