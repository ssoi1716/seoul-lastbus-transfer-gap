import numpy as np
import pandas as pd
from common_direction import haversine_m

def coverage(p_lat, p_lon, s_lat, s_lon, radius):
    if len(p_lat) == 0 or len(s_lat) == 0:
        return np.zeros(len(p_lat), bool)
    d = haversine_m(np.asarray(p_lat)[:, None], np.asarray(p_lon)[:, None], np.asarray(s_lat)[None, :], np.asarray(s_lon)[None, :])
    return (d <= radius).any(axis=1)

def build_demand_points(rows):
    recs = []
    for r in rows:
        n = len(r["lat"])
        if n == 0 or r["weight"] <= 0:
            continue
        for la, lo in zip(r["lat"], r["lon"]):
            recs.append({"id": r["id"], "lat": la, "lon": lo, "w": r["weight"] / n})
    return pd.DataFrame(recs)

def greedy_max_cover(points, baseline, candidates, radius=400.0, stop_gain=0.05, max_pick=None, tie_break=None, curve_extend=12):
    W = points["w"].sum()
    pl, po = points["lat"].values, points["lon"].values
    cov = np.zeros(len(points), bool)
    for la, lo in baseline.values():
        cov |= coverage(pl, po, la, lo, radius)
    base = points["w"][cov].sum() / W
    cmask = {k: coverage(pl, po, la, lo, radius) for k, (la, lo) in candidates.items()}
    tb = tie_break or {}
    remaining, selected, curve = set(candidates), [], [(0, base)]
    while remaining and (max_pick is None or len(selected) < max_pick):
        gains = {k: points["w"][cmask[k] & ~cov].sum() / W for k in remaining}
        best = min(remaining, key=lambda k: (-gains[k], tb.get(k, 0)))
        if gains[best] < stop_gain:
            break
        cov |= cmask[best]; remaining.remove(best)
        selected.append((best, gains[best], points["w"][cov].sum() / W))
        curve.append((len(selected), selected[-1][2]))
    final = points["w"][cov].sum() / W
    tcov, trem, k = cov.copy(), set(remaining), len(selected)
    while trem and k < curve_extend:
        gains = {c: points["w"][cmask[c] & ~tcov].sum() / W for c in trem}
        b = max(gains, key=gains.get)
        if gains[b] <= 0:
            break
        tcov |= cmask[b]; trem.remove(b); k += 1
        curve.append((k, points["w"][tcov].sum() / W))
    return {"selected": selected, "base_cov": base, "final_cov": final, "curve": curve, "covered": cov}

def covered_share_by_row(points, covered):
    df = points.assign(c=covered)
    return df.groupby("id").apply(lambda g: (g["w"] * g["c"]).sum() / g["w"].sum(), include_groups=False)

def assign_action(row_gap, selected_ids, row_id, covered_share, route_link=None, bench=None):
    if row_gap is None or (isinstance(row_gap, float) and np.isnan(row_gap)):
        return "판정 보류"
    if row_gap <= 0:
        return "연결됨(조치 불필요)"
    if row_id in selected_ids:
        return f"막차 {int(np.ceil(row_gap / 5) * 5)}분 연장" if row_gap <= 15 else "막차 연계 1회 추가"
    if covered_share >= 0.8:
        return "다른 노선·올빼미로 대체 연결"
    if covered_share > 0:
        return "일부 대체 연결 + 심야 DRT 검토"
    return "심야 DRT 검토"
