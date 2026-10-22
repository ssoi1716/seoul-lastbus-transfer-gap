import pickle, math, sys
import pandas as pd, numpy as np

R = float(sys.argv[1]) if len(sys.argv) > 1 else 400.0
GAIN_STOP = 0.05
P = pd.read_pickle("out/_stage2.pkl")
aux = pickle.load(open("out/_stage2_aux.pkl", "rb"))
night, st_last = aux["night"], aux["st_last"]
mix = pd.read_csv("out/gap_quadrant_candidates_2025.csv")[["name", "D_median"]].set_index("name")["D_median"]

def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))

def covers(plat, plon, slat, slon):
    if len(slat) == 0 or len(plat) == 0:
        return np.zeros(len(plat), bool)
    d = hav(plat[:, None], plon[:, None], slat[None, :], slon[None, :])
    return (d <= R).any(axis=1)

P = P[P["API 조회"] == "Y"].copy().reset_index(drop=True)
P["dup"] = P.groupby(["역", "정류소ARS", "노선"])["노선"].transform("size")
P["w_row"] = P["22~23시 승차(일평균)"] / P["dup"]
res_rows, curves, summ = [], [], []
for stn, g in P.groupby("역"):
    g = g.reset_index(drop=True)
    pts_lat, pts_lon, pts_w, pts_row = [], [], [], []
    for i, r in g.iterrows():
        n = len(r.down_lat)
        if n == 0:
            continue
        pts_lat += list(r.down_lat); pts_lon += list(r.down_lon); pts_w += [r.w_row / n] * n; pts_row += [i] * n
    plat, plon, pw = np.array(pts_lat), np.array(pts_lon), np.array(pts_w)
    W = pw.sum()
    cov = np.zeros(len(pw), bool)
    for i, r in g[g["부족(분)"] <= 0].iterrows():
        cov |= covers(plat, plon, r.down_lat, r.down_lon)
    base_conn = cov.copy()
    for nm, nla, nlo in night.get(stn, []):
        cov |= covers(plat, plon, nla, nlo)
    base = pw[cov].sum() / W
    curve = [(0, base)]
    chosen = []
    cand = set(g.index[g["부족(분)"] > 0])
    cmask = {i: covers(plat, plon, g.loc[i, "down_lat"], g.loc[i, "down_lon"]) for i in cand}
    while cand:
        gains = {i: pw[cmask[i] & ~cov].sum() / W for i in cand}
        best = max(gains, key=lambda i: (gains[i], -g.loc[i, "부족(분)"]))
        if gains[best] < GAIN_STOP:
            break
        cov |= cmask[best]; cand.remove(best); chosen.append((best, gains[best]))
        curve.append((len(chosen), pw[cov].sum() / W))
    tmp_cov, tmp_cand, k = cov.copy(), set(cand), len(chosen)
    while tmp_cand and k < 12:
        gains = {i: pw[cmask[i] & ~tmp_cov].sum() / W for i in tmp_cand}
        best = max(gains, key=gains.get)
        if gains[best] <= 0:
            break
        tmp_cov |= cmask[best]; tmp_cand.remove(best); k += 1
        curve.append((k, pw[tmp_cov].sum() / W))
    for k, c in curve:
        curves.append({"역": stn, "조정 노선 수": k, "커버율": c, "선택 구간": k <= len(chosen)})
    final = pw[cov].sum() / W
    chosen_idx = {i for i, _ in chosen}
    for i, r in g.iterrows():
        if r["부족(분)"] <= 0:
            act = "연결됨(조치 불필요)"
        elif i in chosen_idx:
            act = f"막차 {int(math.ceil(r['부족(분)'] / 5) * 5)}분 연장" if r["부족(분)"] <= 15 else "막차 연계 1회 추가"
        else:
            m = np.array(pts_row) == i
            share = cov[m].mean() if m.any() else 0.0
            act = "다른 노선·올빼미로 대체 연결" if share >= 0.8 else ("일부 대체 연결 + 심야 DRT 검토" if share > 0 else "심야 DRT 검토")
        gain = dict(chosen).get(i, np.nan)
        res_rows.append({**r.drop(labels=["down_lat", "down_lon", "막차_분", "dup", "w_row"]).to_dict(), "최소 조정안": act,
                         "선택 순서": ([j for j, _ in chosen].index(i) + 1) if i in chosen_idx else None, "커버 증가(%p)": round(gain * 100, 1) if gain == gain else None})
    n_ext = sum(1 for i, _ in chosen if g.loc[i, "부족(분)"] <= 15)
    summ.append({"역": stn, "귀가 노선-방향": len(g), "기존 연결 노선": int((g["부족(분)"] <= 0).sum()), "올빼미 노선": len({n for n, _, _ in night.get(stn, [])}),
                 "조정 전 커버율": round(base * 100, 1), "조정 노선 수": len(chosen), "연장": n_ext, "1회 추가": len(chosen) - n_ext,
                 "조정 후 커버율": round(final * 100, 1), "22~23시 승차 합(일평균)": round(W, 1),
                 "심야 하차 중앙값": float(mix.get(stn, np.nan)), "연결 증가 대리지표(명/평일)": round(float(mix.get(stn, np.nan)) * (final - base), 0)})
S = pd.DataFrame(summ).sort_values("조정 후 커버율")
X = pd.DataFrame(res_rows)
C = pd.DataFrame(curves)
tag = "" if R == 400 else f"_R{int(R)}"
with pd.ExcelWriter(f"out/2단계_역별_최소조정안{tag}.xlsx") as w:
    S.to_excel(w, sheet_name="역별 요약", index=False)
    X.sort_values(["역", "선택 순서"]).to_excel(w, sheet_name="노선별 조치", index=False)
    C.to_excel(w, sheet_name="커버 곡선", index=False)
pd.to_pickle({"S": S, "X": X, "C": C, "R": R}, f"out/_mclp{tag}.pkl")
print(f"[R={R:.0f}m] 조정 노선 합계 {S['조정 노선 수'].sum()} (연장 {S['연장'].sum()}, 1회 추가 {S['1회 추가'].sum()})"
      f" | 커버율 가중평균 {np.average(S['조정 전 커버율'], weights=S['22~23시 승차 합(일평균)']):.1f}% → {np.average(S['조정 후 커버율'], weights=S['22~23시 승차 합(일평균)']):.1f}%"
      f" | 연결 증가 대리지표 {S['연결 증가 대리지표(명/평일)'].sum():,.0f}명/평일")
print(S.to_string(index=False))
print("조치 분포:", X["최소 조정안"].str.replace(r"막차 \d+분 연장", "막차 연장", regex=True).value_counts().to_dict())
