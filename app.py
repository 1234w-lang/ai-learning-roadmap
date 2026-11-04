import streamlit as st
import pandas as pd
import joblib

# 1. 设置网页标题
st.set_page_config(page_title="共享单车需求预测", page_icon="🚲")
st.title("🚲 共享单车需求预测系统")
st.write("通过调整下方的天气和时间参数，模型将实时预测该时段的租车需求量。")

# 2. 加载训练好的模型
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

model = load_model()

# 3. 构建侧边栏输入
st.sidebar.header("输入特征参数")
hr = st.sidebar.slider("小时 (0-23)", 0, 23, 8)
temp = st.sidebar.slider("温度 (归一化 0-1)", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("湿度 (归一化 0-1)", 0.0, 1.0, 0.5)
weathersit = st.sidebar.selectbox("天气状况", [1, 2, 3], format_func=lambda x: {1:"晴天", 2:"多云", 3:"雨雪"}[x])
workingday = st.sidebar.selectbox("是否工作日", [1, 0], format_func=lambda x: {1:"是", 0:"否"}[x])

# 这里为了简单，其他特征填默认值
input_data = pd.DataFrame({
    'season': [1], 'yr': [1], 'mnth': [5], 'hr': [hr],
    'holiday': [0], 'weekday': [1], 'workingday': [workingday],
    'weathersit': [weathersit], 'temp': [temp], 'atemp': [temp],
    'hum': [hum], 'windspeed': [0.2]
})

# 4. 预测并展示
if st.button("开始预测"):
    prediction = model.predict(input_data)[0]
    st.success(f"预测结果：该时段大约会有 **{int(prediction)}** 辆单车被租借。")
    st.info("💡 业务建议：根据预测结果，可提前调度车辆至该区域。")