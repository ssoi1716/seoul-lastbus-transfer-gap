import re
from dataclasses import dataclass
import numpy as np
import pandas as pd

def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))

def _nz(s):
    return re.sub(r"[\s\.\·\-\(\)]", "", str(s))

def ars5(v):
    return str(v).replace(".0", "").strip().zfill(5)

def load_sequence(path):
    df = pd.read_excel(path, dtype={"ARS_ID": str, "노선명": str})
    df = df.rename(columns={"ROUTE_ID": "route_id", "노선명": "route_name", "순번": "seq",
                            "정류소명": "stop_name", "X좌표": "lon", "Y좌표": "lat"})
    df["ars"] = df["ARS_ID"].map(ars5)
    return df[["route_id", "route_name", "seq", "ars", "stop_name", "lat", "lon"]].sort_values(["route_id", "seq"]).reset_index(drop=True)

def load_basic_info(path, sheet="평일공동배차"):
    bi = pd.read_excel(path, sheet_name=sheet, dtype={"노선번호": str})
    return {r["노선번호"]: (r["기점명"], r["종점명"]) for _, r in bi.drop_duplicates("노선번호").iterrows()}

def load_stops(path):
    st = pd.read_excel(path, dtype={"ARS_ID": str})
    st["ars"] = st["ARS_ID"].map(ars5)
    return st.drop_duplicates("ars").set_index("ars").rename(columns={"정류소명": "stop_name", "X좌표": "lon", "Y좌표": "lat"})[["stop_name", "lat", "lon"]]

@dataclass
class RouteLegs:
    route_id: int
    route_name: str
    stops: pd.DataFrame
    turn_idx: int
    method: str
    circular: bool

def find_turning_point(stops, terminal_name=None):
    d = haversine_m(stops["lat"].iloc[0], stops["lon"].iloc[0], stops["lat"].values, stops["lon"].values)
    if terminal_name:
        key = _nz(terminal_name)
        matches = [i for i, n in enumerate(stops["stop_name"]) if i > 0 and key and (key in _nz(n) or _nz(n) in key)]
        if matches:
            return max(matches, key=lambda i: d[i]), "종점명 일치"
    return int(np.argmax(d)), "최원점 근사"

def build_legs(seq, basic_info):
    legs = {}
    for rid, g in seq.groupby("route_id"):
        g = g.reset_index(drop=True)
        name = g["route_name"].iloc[0]
        term = basic_info.get(name, (None, None))[1]
        idx, method = find_turning_point(g, term)
        o, t = basic_info.get(name, (None, None))
        circ = bool(o is not None and t is not None and _nz(o) == _nz(t))
        legs[rid] = RouteLegs(rid, name, g, idx, method, circ)
    return legs

def stop_directions(leg, ars):
    g, turn = leg.stops, leg.turn_idx
    out = []
    for i in np.where(g["ars"].values == ars)[0]:
        if i < turn:
            dir_no, end, dest, label = 1, turn, g["stop_name"].iloc[turn], "기점→종점"
        elif i == turn:
            dir_no, end, dest, label = 2, len(g) - 1, g["stop_name"].iloc[-1], "회차점→기점"
        else:
            dir_no, end, dest, label = 2, len(g) - 1, g["stop_name"].iloc[-1], "종점→기점"
        down = g.iloc[i + 1:end + 1]
        out.append({"seq": int(g["seq"].iloc[i]), "dir_no": dir_no, "dir_label": label, "dest": dest,
                    "downstream": down, "is_terminal": len(down) == 0})
    return out

def downstream_coords(leg, ars, dir_no):
    for d in stop_directions(leg, ars):
        if d["dir_no"] == dir_no:
            return d["downstream"]["lat"].values, d["downstream"]["lon"].values
    return np.array([]), np.array([])

def station_directions(station, st_lat, st_lon, stops_xy, seq, legs, radius=300, route_names=None):
    near = stops_xy[haversine_m(st_lat, st_lon, stops_xy["lat"].values, stops_xy["lon"].values) <= radius]
    rows = []
    for ars, s in near.iterrows():
        dist = float(haversine_m(st_lat, st_lon, s["lat"], s["lon"]))
        for rid in seq.loc[seq["ars"] == ars, "route_id"].unique():
            leg = legs[rid]
            if route_names is not None and leg.route_name not in route_names:
                continue
            for d in stop_directions(leg, ars):
                rows.append({"역": station, "노선": leg.route_name, "노선ID": rid, "정류소ARS": ars, "정류소명": s["stop_name"],
                             "역_정류소_거리m": round(dist), "순번": d["seq"], "방향번호": d["dir_no"], "방향구간": d["dir_label"],
                             "방면": d["dest"], "하류정류소수": len(d["downstream"]), "종점정류소": d["is_terminal"],
                             "회차점판정": leg.method, "회차점순번": int(leg.stops["seq"].iloc[leg.turn_idx]), "순환노선": leg.circular})
    return pd.DataFrame(rows).sort_values(["노선", "정류소ARS", "방향번호"]).reset_index(drop=True)
