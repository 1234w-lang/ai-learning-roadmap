import os
import pandas as pd

def load_data(filepath):
    """读取 CSV 文件，并包含异常处理"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ 成功读取文件: {filepath}")
        return df
    except FileNotFoundError:
        print(f"❌ 找不到文件: {filepath}，请检查路径是否正确。")
        return None
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        return None

def summarize_data(df, name):
    """输出数据摘要：行数、列名、缺失值、时间范围"""
    if df is None:
        return
    
    print(f"\n--- {name} 数据摘要 ---")
    print(f"数据行数: {len(df)} 行")
    print(f"数据列数: {len(df.columns)} 列")
    
    # 查看缺失值
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("缺失值: 无")
    else:
        print("缺失值统计:")
        print(missing[missing > 0])
        
    # 查看时间范围 (针对 bike sharing 的 dteday 字段)
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday']) # 转成时间格式
        print(f"时间范围: {df['dteday'].min().date()} 至 {df['dteday'].max().date()}")

if __name__ == "__main__":
    # 设置文件路径
    hour_path = os.path.join("data", "raw", "hour.csv")
    day_path = os.path.join("data", "raw", "day.csv")
    
    # 读取并输出摘要
    hour_df = load_data(hour_path)
    summarize_data(hour_df, "hour.csv")
    
    day_df = load_data(day_path)
    summarize_data(day_df, "day.csv")