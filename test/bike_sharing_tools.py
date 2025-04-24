import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import os
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

from langgraph.prebuilt import ToolNode

@tool
def query_station_euler_24h(user_input: str, filepath: str = "data/bike_data.xlsx") -> str:
    """Get the bike count at a specific station over 24 hours using Euler simulation."""
    messages = ""

    # 读取数据
    data = pd.read_excel(filepath)
    messages += f"Loaded data from '{filepath}', total records: {len(data)}.\n"

    # 提取站点信息
    id_name_pairs = pd.concat([
        data[["start_station_id", "start_station_name"]].rename(columns={"start_station_id": "station_id", "start_station_name": "station_name"}),
        data[["end_station_id", "end_station_name"]].rename(columns={"end_station_id": "station_id", "end_station_name": "station_name"})
    ]).dropna().drop_duplicates()
    id_name_pairs["station_id"] = id_name_pairs["station_id"].astype(str)
    id_name_pairs["station_name"] = id_name_pairs["station_name"].astype(str)

    name_to_id = id_name_pairs.set_index("station_name")["station_id"].to_dict()
    id_to_name = id_name_pairs.set_index("station_id")["station_name"].to_dict()

    # 解析输入
    if user_input in id_to_name:
        station_id = user_input
        station_name = id_to_name[station_id]
    elif user_input in name_to_id:
        station_name = user_input
        station_id = name_to_id[station_name]
    else:
        return f"输入 '{user_input}' 既不是有效的站点名，也不是有效的站点 ID。"

    messages += f"Target station: {station_name} (ID: {station_id}).\n"

    # 构建站点映射
    stations = pd.unique(data[["start_station_id", "end_station_id"]].values.ravel())
    stations = np.sort([str(s) for s in stations if pd.notna(s)])
    station_map = {station: idx for idx, station in enumerate(stations)}
    n = len(stations)

    if station_id not in station_map:
        return f"Station ID '{station_id}' not found in station map."

    station_idx = station_map[station_id]
    messages += f"Station index in OD matrix: {station_idx}, total stations: {n}.\n"

    # 构建 OD 矩阵
    OD = np.zeros((n, n))
    for _, row in data.iterrows():
        if pd.isna(row.start_station_id) or pd.isna(row.end_station_id):
            continue
        s = station_map[str(row.start_station_id)]
        e = station_map[str(row.end_station_id)]
        OD[s, e] += 1

    inflow = OD.sum(axis=0)
    outflow = OD.sum(axis=1)
    delta = inflow - outflow
    messages += f"Computed inflow/outflow vectors.\n"

    # 优化调度
    def objective(x): return 0.5 * np.dot(x, x) + np.dot(delta, x)
    res = minimize(objective, np.zeros(n), bounds=Bounds([-np.inf]*n, [np.inf]*n))
    x_opt = res.x
    messages += f"Optimization finished. Objective value: {res.fun:.2f}.\n"

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

    messages += f"Euler simulation completed for {steps} steps.\n"

    station_data = x_euler[station_idx]
    messages += f"Final simulated data for station '{station_name}' (ID: {station_id}):\n"
    messages += ", ".join([f"{v:.2f}" for v in station_data]) + "\n"

    return messages

# ✅ 示例调用（取消注释运行）
# query_station_euler_24h("data/bike_data.xlsx", "30201")  # 用 ID
# query_station_euler_24h("data/bike_data.xlsx", "9th & G St NW")  # 用名称


tools = [query_station_euler_24h]
tool_node = ToolNode(tools)

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "query_station_euler_24h",
            "args": {"user_input": "9th & G St NW", "filepath": "data/bike_data.xlsx"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

print(tool_node.invoke({"messages": [message_with_single_tool_call]}))
