from langchain_core.tools import BaseTool
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
import asyncio
from typing import Any, Callable, List, Optional, cast, Union

# 参数输入格式
class StationInput(BaseModel):
    user_input: str = Field(..., description="站点名或站点ID")
    filepath: str = Field(default="data/bike_data.xlsx", description="数据文件路径")

class QueryStationEuler24hTool(BaseTool):
    name: str = "QueryStationEuler24hTool"
    description: str = "使用欧拉法估算一个车站未来24小时内的车辆数量。"
    args_schema: type = StationInput
    response_format: str = "content_and_artifact"

    def _run(self,
                    user_input: str,
                    filepath: str = "data/bike_data.xlsx",
                    run_manager: Optional[Any] = None) -> Tuple[str, List[float]]:

        messages = ""
        try:
            # data = await asyncio.to_thread(pd.read_excel, filepath)
            data = pd.read_excel(filepath)
            messages += f"Loaded data from '{filepath}', total records: {len(data)}.\n"

            # 提取站点映射
            id_name_pairs = pd.concat([
                data[["start_station_id", "start_station_name"]].rename(columns={"start_station_id": "station_id", "start_station_name": "station_name"}),
                data[["end_station_id", "end_station_name"]].rename(columns={"end_station_id": "station_id", "end_station_name": "station_name"})
            ]).dropna().drop_duplicates()

            id_name_pairs["station_id"] = id_name_pairs["station_id"].astype(str)
            id_name_pairs["station_name"] = id_name_pairs["station_name"].astype(str)

            name_to_id = id_name_pairs.set_index("station_name")["station_id"].to_dict()
            id_to_name = id_name_pairs.set_index("station_id")["station_name"].to_dict()

            if user_input in id_to_name:
                station_id = user_input
                station_name = id_to_name[station_id]
            elif user_input in name_to_id:
                station_name = user_input
                station_id = name_to_id[station_name]
            else:
                return f"输入 '{user_input}' 无法识别为合法站点名或ID", []

            messages += f"Target station: {station_name} (ID: {station_id}).\n"

            stations = pd.unique(data[["start_station_id", "end_station_id"]].values.ravel())
            stations = np.sort([str(s) for s in stations if pd.notna(s)])
            station_map = {station: idx for idx, station in enumerate(stations)}
            n = len(stations)

            if station_id not in station_map:
                return f"Station ID '{station_id}' not found in station map.", []

            station_idx = station_map[station_id]
            messages += f"Station index in OD matrix: {station_idx}, total stations: {n}.\n"

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

            def solve_optimization():
                def objective(x): return 0.5 * np.dot(x, x) + np.dot(delta, x)
                bounds = Bounds([-np.inf] * n, [np.inf] * n)
                return minimize(objective, np.zeros(n), bounds=bounds)

            res = solve_optimization()
            x_opt = res.x
            messages += f"Optimization finished. Objective value: {res.fun:.2f}.\n"

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

            station_data = x_euler[station_idx]
            messages += f"Final simulated data for station '{station_name}' (ID: {station_id}):\n"
            messages += ", ".join([f"{v:.2f}" for v in station_data]) + "\n"

            return messages, station_data.tolist()

        except Exception as e:
            return f"❌ Error: {str(e)}", []
        
class QueryStationRK24hTool(BaseTool):
    name: str = "QueryStationRK24hTool"
    description: str = "使用Runge-Kutta四阶法估算一个车站未来24小时内的车辆数量。"
    args_schema: type = StationInput
    response_format: str = "content_and_artifact"

    def _run(self,
                    user_input: str,
                    filepath: str = "data/bike_data.xlsx",
                    run_manager: Optional[Any] = None) -> Tuple[str, List[float]]:

        messages = ""
        try:
            data = pd.read_excel(filepath)
            messages += f"Loaded data from '{filepath}', total records: {len(data)}.\n"

            id_name_pairs = pd.concat([
                data[["start_station_id", "start_station_name"]].rename(columns={"start_station_id": "station_id", "start_station_name": "station_name"}),
                data[["end_station_id", "end_station_name"]].rename(columns={"end_station_id": "station_id", "end_station_name": "station_name"})
            ]).dropna().drop_duplicates()

            id_name_pairs["station_id"] = id_name_pairs["station_id"].astype(str)
            id_name_pairs["station_name"] = id_name_pairs["station_name"].astype(str)

            name_to_id = id_name_pairs.set_index("station_name")["station_id"].to_dict()
            id_to_name = id_name_pairs.set_index("station_id")["station_name"].to_dict()

            if user_input in id_to_name:
                station_id = user_input
                station_name = id_to_name[station_id]
            elif user_input in name_to_id:
                station_name = user_input
                station_id = name_to_id[station_name]
            else:
                return f"输入 '{user_input}' 无法识别为合法站点名或ID", []

            messages += f"Target station: {station_name} (ID: {station_id}).\n"

            stations = pd.unique(data[["start_station_id", "end_station_id"]].values.ravel())
            stations = np.sort([str(s) for s in stations if pd.notna(s)])
            station_map = {station: idx for idx, station in enumerate(stations)}
            n = len(stations)

            if station_id not in station_map:
                return f"Station ID '{station_id}' not found in station map.", []

            station_idx = station_map[station_id]
            messages += f"Station index in OD matrix: {station_idx}, total stations: {n}.\n"

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

            def solve_optimization():
                def objective(x): return 0.5 * np.dot(x, x) + np.dot(delta, x)
                bounds = Bounds([-np.inf] * n, [np.inf] * n)
                return minimize(objective, np.zeros(n), bounds=bounds)

            res = solve_optimization()
            x_opt = res.x
            messages += f"Optimization finished. Objective value: {res.fun:.2f}.\n"

            row_sums = OD.sum(axis=1, keepdims=True)
            T = np.divide(OD, row_sums, where=row_sums != 0)

            def bike_ode(x, T, u):
                inflow = T.T @ x
                outflow = T.sum(axis=1) * x
                return inflow - outflow + u

            h = 1
            T_total = 24
            steps = int(T_total / h)
            x_rk = np.zeros((n, steps + 1))
            x_rk[:, 0] = inflow
            u = x_opt

            for k in range(steps):
                k1 = bike_ode(x_rk[:, k], T, u)
                k2 = bike_ode(x_rk[:, k] + h/2 * k1, T, u)
                k3 = bike_ode(x_rk[:, k] + h/2 * k2, T, u)
                k4 = bike_ode(x_rk[:, k] + h * k3, T, u)
                x_rk[:, k + 1] = x_rk[:, k] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

            station_data = x_rk[station_idx]
            messages += f"Final simulated data for station '{station_name}' (ID: {station_id}) using Runge-Kutta:\n"
            messages += ", ".join([f"{v:.2f}" for v in station_data]) + "\n"

            return messages, station_data.tolist()

        except Exception as e:
            return f"❌ Error: {str(e)}", []

TOOLS: List[Callable[..., Any]] = [QueryStationEuler24hTool(), QueryStationRK24hTool()]