import os, sys, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from common_mclp import build_demand_points, greedy_max_cover, covered_share_by_row

mk = lambda lat0, n: (np.full(n, lat0) + np.arange(n) * 0.0005, np.full(n, 127.0))
rows = [{"id": "A", "weight": 10, "lat": mk(37.50, 4)[0], "lon": mk(37.50, 4)[1]},
        {"id": "B", "weight": 3, "lat": mk(37.52, 4)[0], "lon": mk(37.52, 4)[1]},
        {"id": "C", "weight": 1, "lat": mk(37.54, 4)[0], "lon": mk(37.54, 4)[1]}]
pts = build_demand_points(rows)
cands = {r["id"]: (r["lat"], r["lon"]) for r in rows}
res = greedy_max_cover(pts, baseline={}, candidates=cands, radius=400, stop_gain=0.10)
assert [s[0] for s in res["selected"]] == ["A", "B"], res["selected"]
assert abs(res["final_cov"] - 13 / 14) < 1e-9
res2 = greedy_max_cover(pts, baseline={"N": cands["A"]}, candidates=cands, radius=400, stop_gain=0.05)
assert abs(res2["base_cov"] - 10 / 14) < 1e-9 and [s[0] for s in res2["selected"]] == ["B", "C"]
print("1) 장난감 예제 통과: 선택 순서 A→B, 기존 서비스 반영 시 B→C")

P = pd.read_pickle("out/_stage2.pkl")
aux = pickle.load(open("out/_stage2_aux.pkl", "rb"))
P = P[P["API 조회"] == "Y"].copy()
P["dup"] = P.groupby(["역", "정류소ARS", "노선"])["노선"].transform("size")
P["w"] = P["22~23시 승차(일평균)"] / P["dup"]
expect = {"사당": (0.472, 0.846, 3), "강변": (0.168, 1.000, 2)}
for stn, (b0, b1, k) in expect.items():
    g = P[P["역"] == stn].reset_index(drop=True)
    rows = [{"id": i, "weight": r.w, "lat": r.down_lat, "lon": r.down_lon} for i, r in g.iterrows()]
    pts = build_demand_points(rows)
    base = {f"conn{i}": (r.down_lat, r.down_lon) for i, r in g[g["부족(분)"] <= 0].iterrows()}
    base.update({f"N{j}": (la, lo) for j, (_, la, lo) in enumerate(aux["night"].get(stn, []))})
    cand = {i: (r.down_lat, r.down_lon) for i, r in g[g["부족(분)"] > 0].iterrows()}
    tb = {i: r["부족(분)"] for i, r in g.iterrows()}
    res = greedy_max_cover(pts, base, cand, radius=400, stop_gain=0.05, tie_break=tb)
    print(f"2) {stn}: 커버 {res['base_cov']:.3f} → {res['final_cov']:.3f}, 선택 {len(res['selected'])}개 (기대 {b0}→{b1}, {k}개)")
    assert abs(res["base_cov"] - b0) < 0.002 and abs(res["final_cov"] - b1) < 0.002 and len(res["selected"]) == k
    share = covered_share_by_row(pts, res["covered"])
    print("   노선-방향별 커버 비율 예:", share.round(2).head(3).to_dict())
print("모든 검증 통과")
