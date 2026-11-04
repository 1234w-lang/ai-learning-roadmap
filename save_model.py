import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# 读数据
df = pd.read_csv("data/raw/hour.csv")
df['dteday'] = pd.to_datetime(df['dteday'])
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
X = df[features]
y = df['cnt']

# 为了快，用全量数据训一个最终模型
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# 保存模型文件
joblib.dump(rf, "rf_model.pkl")
print("模型已保存为 rf_model.pkl")