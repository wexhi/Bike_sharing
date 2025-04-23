import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize, Bounds
import os

# 创建保存目录
os.makedirs("res", exist_ok=True)
os.makedirs("images", exist_ok=True)

# 读取数据
data = pd.read_excel("data/bike_data.xlsx")
stations = np.sort(pd.unique(data[["start_station_id", "end_station_id"]].values.ravel()))

n = len(stations)
station_map = {station: idx for idx, station in enumerate(stations)}

# 构建 OD 矩阵
OD = np.zeros((n, n))
for _, row in data.iterrows():
    s = station_map[row.start_station_id]
    e = station_map[row.end_station_id]
    OD[s, e] += 1

# 可视化 OD 矩阵
plt.figure()
plt.imshow(OD, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('OD Demand Matrix')
plt.xlabel('Destination Station')
plt.ylabel('Origin Station')
plt.tight_layout()
plt.savefig("images/od_matrix.png")

# 计算 inflow, outflow 和 delta
inflow = OD.sum(axis=0)
outflow = OD.sum(axis=1)
delta = inflow - outflow

# 二次规划求解调度变量 x_opt
H = np.eye(n)
f = delta
def objective(x):
    return 0.5 * np.dot(x, x) + np.dot(f, x)

bounds = Bounds([-np.inf]*n, [np.inf]*n)
res = minimize(objective, np.zeros(n), bounds=bounds)
x_opt = res.x

# 可视化调度建议
plt.figure()
plt.bar(np.arange(n), x_opt)
plt.title("Bike Dispatch Recommendation")
plt.xlabel("Station Index")
plt.ylabel("Adjustment Quantity")
plt.tight_layout()
plt.savefig("images/bike_schedule_bar.png")

# 构建迁移概率矩阵 T（按行归一化）
row_sums = OD.sum(axis=1, keepdims=True)
T = np.divide(OD, row_sums, where=row_sums != 0)

# 动态系统函数定义
def bike_ode(x, T, u):
    inflow = T.T @ x
    outflow = T.sum(axis=1) * x
    return inflow - outflow + u

# 模拟参数
h = 1
T_total = 24
steps = int(T_total / h)
x_euler = np.zeros((n, steps + 1))
x_euler[:, 0] = inflow
u = x_opt

# Euler 模拟
for k in range(steps):
    dx = bike_ode(x_euler[:, k], T, u)
    x_euler[:, k + 1] = x_euler[:, k] + h * dx

# Runge-Kutta 四阶模拟
x_rk = np.zeros((n, steps + 1))
x_rk[:, 0] = inflow
for k in range(steps):
    k1 = bike_ode(x_rk[:, k], T, u)
    k2 = bike_ode(x_rk[:, k] + h/2 * k1, T, u)
    k3 = bike_ode(x_rk[:, k] + h/2 * k2, T, u)
    k4 = bike_ode(x_rk[:, k] + h * k3, T, u)
    x_rk[:, k + 1] = x_rk[:, k] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

# 可视化 Euler 模拟结果
plt.figure()
for i in range(min(5, n)):
    plt.plot(np.arange(0, T_total + h, h), x_euler[i, :], label=f"Station {i+1}")
plt.title('Euler Simulation of Bike Flow')
plt.xlabel('Time (hours)')
plt.ylabel('Number of Bikes')
plt.legend()
plt.tight_layout()
plt.savefig("images/euler_simulation.png")

# 可视化 Runge-Kutta 模拟结果
plt.figure()
for i in range(min(5, n)):
    plt.plot(np.arange(0, T_total + h, h), x_rk[i, :], label=f"Station {i+1}")
plt.title('Runge-Kutta 4th Order Simulation')
plt.xlabel('Time (hours)')
plt.ylabel('Number of Bikes')
plt.legend()
plt.tight_layout()
plt.savefig("images/rk_simulation.png")

# 保存结果为 CSV 文件
schedule_result = pd.DataFrame({"StationID": np.arange(1, n+1), "BikeAdjustment": x_opt})
schedule_result.to_csv("res/bike_schedule.csv", index=False)

euler_result = pd.DataFrame(x_euler.T, columns=[f"Station{i+1}" for i in range(n)])
euler_result["Time"] = np.arange(0, T_total+h, h)
euler_result.to_csv("res/bike_dynamic_euler.csv", index=False)

rk_result = pd.DataFrame(x_rk.T, columns=[f"Station{i+1}" for i in range(n)])
rk_result["Time"] = np.arange(0, T_total+h, h)
rk_result.to_csv("res/bike_dynamic_rk.csv", index=False)
