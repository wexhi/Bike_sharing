import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import os

def query_station_euler_24h(filepath: str, user_input: str):
    # 读取数据
    data = pd.read_excel(filepath)

    # 提取所有站点 ID 和 名称
    id_name_pairs = pd.concat([
        data[["start_station_id", "start_station_name"]].rename(columns={"start_station_id": "station_id", "start_station_name": "station_name"}),
        data[["end_station_id", "end_station_name"]].rename(columns={"end_station_id": "station_id", "end_station_name": "station_name"})
    ]).dropna().drop_duplicates()

    # 转为字符串处理（ID 可能是 float）
    id_name_pairs["station_id"] = id_name_pairs["station_id"].astype(str)
    id_name_pairs["station_name"] = id_name_pairs["station_name"].astype(str)

    # 建立 name → id 映射 和 id → name 映射
    name_to_id = id_name_pairs.set_index("station_name")["station_id"].to_dict()
    id_to_name = id_name_pairs.set_index("station_id")["station_name"].to_dict()

    # 判断输入
    if user_input in id_to_name:
        station_id = user_input
        station_name = id_to_name[station_id]
    elif user_input in name_to_id:
        station_name = user_input
        station_id = name_to_id[station_name]
    else:
        raise ValueError(f"输入 '{user_input}' 既不是有效的站点名，也不是有效的站点 ID。")

    # 构建站点索引表
    stations = pd.unique(data[["start_station_id", "end_station_id"]].values.ravel())
    stations = np.sort([str(s) for s in stations if pd.notna(s)])
    station_map = {station: idx for idx, station in enumerate(stations)}
    n = len(stations)

    if station_id not in station_map:
        raise ValueError(f"Station ID '{station_id}' not found in station map.")

    station_idx = station_map[station_id]

    # 构建 OD 矩阵
    OD = np.zeros((n, n))
    for _, row in data.iterrows():
        if pd.isna(row.start_station_id) or pd.isna(row.end_station_id):
            continue
        s = station_map[str(row.start_station_id)]
        e = station_map[str(row.end_station_id)]
        OD[s, e] += 1

    # inflow, outflow, delta
    inflow = OD.sum(axis=0)
    outflow = OD.sum(axis=1)
    delta = inflow - outflow

    # 调度求解
    def objective(x): return 0.5 * np.dot(x, x) + np.dot(delta, x)
    bounds = Bounds([-np.inf]*n, [np.inf]*n)
    res = minimize(objective, np.zeros(n), bounds=bounds)
    x_opt = res.x

    # 构建转移矩阵
    row_sums = OD.sum(axis=1, keepdims=True)
    T = np.divide(OD, row_sums, where=row_sums != 0)

    # Euler 模拟
    h = 1
    T_total = 24
    steps = int(T_total / h)
    x_euler = np.zeros((n, steps + 1))
    x_euler[:, 0] = inflow
    u = x_opt

    def bike_ode(x, T, u):
        inflow = T.T @ x
        outflow = T.sum(axis=1) * x
        return inflow - outflow + u

    for k in range(steps):
        dx = bike_ode(x_euler[:, k], T, u)
        x_euler[:, k + 1] = x_euler[:, k] + h * dx

    # 获取车站模拟数据
    station_data = x_euler[station_idx]

    # 可视化
    plt.plot(np.arange(0, T_total + h, h), station_data, marker='o')
    plt.title(f"Euler Simulation - {station_name}")
    plt.xlabel("Hour")
    plt.ylabel("Bike Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 打印
    print(f"Bike counts at station '{station_name}' (ID: {station_id}) over 24 hours:")
    print(station_data)

# ✅ 示例调用（取消注释运行）
# query_station_euler_24h("data/bike_data.xlsx", "30201")  # 用 ID
query_station_euler_24h("data/bike_data.xlsx", "9th & G St NW")  # 用名称
