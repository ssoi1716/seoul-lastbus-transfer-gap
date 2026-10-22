import os, sys, re, json, glob, calendar, pickle, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team"))
from common_direction import load_sequence, load_stops, build_legs, station_directions, haversine_m
from common_mclp import build_demand_points, greedy_max_cover, covered_share_by_row, coverage

D = "data/"
STATIONS = ["사당", "강변"]
RADIUS_ST, RADIUS_COVER, STOP_GAIN = 300, 400, 0.05
SPEED, PER_STOP = 0.85, 1.45
TRANSFER = 6.0
os.makedirs("out/final", exist_ok=True)

def tm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    p = [int(x) for x in str(s).strip().split(":")]
    if len(p) < 2:
        return np.nan
    h = p[0] + 24 if p[0] < 4 else p[0]
    return h * 60 + p[1] + (p[2] / 60 if len(p) > 2 else 0)

def hhmm(x):
    return "" if x is None or pd.isna(x) else f"{int(x // 60) % 24:02d}:{int(x % 60):02d}"

def rd(p, **kw):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(p, encoding=enc, **kw)
        except UnicodeDecodeError:
            pass

seq = load_sequence(D + "버스_노선별경유정류소순서_20251209.xlsx")
bi = pd.read_excel(D + "버스_노선기본정보_기점첫막차_20251209.xlsx", sheet_name="평일공동배차", dtype={"노선번호": str}).drop_duplicates("노선번호").set_index("노선번호")
binfo = {k: (r["기점명"], r["종점명"]) for k, r in bi.iterrows()}
stops_xy = load_stops(D + "버스_정류소위치_20251209.xlsx")
legs = build_legs(seq, binfo)
c18 = pd.read_csv(D + "지하철_역좌표_1-8호선_20250814.csv", encoding="cp949")
st_xy = c18[c18["역명"].isin(STATIONS)].groupby("역명")[["위도", "경도"]].mean()
R = pd.concat([station_directions(s, st_xy.loc[s, "위도"], st_xy.loc[s, "경도"], stops_xy, seq, legs, radius=RADIUS_ST) for s in STATIONS], ignore_index=True)
home = pd.read_excel("out/8개역_귀가방향노선_2025월별판정.xlsx", dtype={"정류소ARS": str, "노선": str})
hk = set(zip(home["역"], home["노선"], home["정류소ARS"].str.zfill(5)))
R["귀가방향"] = [(a, b, c) in hk for a, b, c in zip(R["역"], R["노선"], R["정류소ARS"])]
R["유형"] = R["노선"].map(lambda r: bi["유형"].get(r, ""))

R["총정류소수"] = R["노선ID"].map(lambda rid: len(legs[rid].stops))
R["기점막차"] = R["노선"].map(lambda r: bi["막차시간"].get(r))
R["운행시간"] = R["노선"].map(lambda r: bi["운행시간"].get(r))
R["배차간격"] = R["노선"].map(lambda r: bi["배차간격"].get(r))
R["origin"] = R["기점막차"].map(tm)

def model_time(row, speed=SPEED):
    if pd.isna(row["origin"]):
        return np.nan
    el = row["순번"] / row["총정류소수"] * row["운행시간"] * speed if pd.notna(row["운행시간"]) else row["순번"] * PER_STOP * (speed / SPEED)
    return np.floor(row["origin"] + el)

R["model"] = R.apply(model_time, axis=1)

K = D + "팀산출물/"
team = pd.read_excel(K + "02_2025_사당강변_정류소별막차_실측전체대조표.xlsx", dtype={"노선": str})
team["ars"] = team["정류소ARS"].astype(int).astype(str).str.zfill(5)
team["team_model"] = team["2025_산출막차시각"].map(tm)
R = R.merge(team[["역", "노선", "ars", "순번", "team_model"]].rename(columns={"ars": "정류소ARS"}), on=["역", "노선", "정류소ARS", "순번"], how="left")
R["repro_diff"] = (R["model"].round() - R["team_model"]).abs()

def api_last(ars, rid):
    f = f"api/cache/bustime_{ars}_{rid}.json"
    if not os.path.exists(f):
        return np.nan
    j = json.load(open(f, encoding="utf-8"))
    v = re.sub(r"\D", "", str(j.get("item", {}).get("lastBusTm", "")))
    if not j.get("ok") or len(v) < 4:
        return np.nan
    h = int(v[:2]); h = h + 24 if h < 4 else h
    return float(h * 60 + int(v[2:4]))

R["api"] = [api_last(a, r) for a, r in zip(R["정류소ARS"], R["노선ID"])]
R["err"] = R["model"] - R["api"]

def down_ars(row):
    g, turn = legs[row["노선ID"]].stops, legs[row["노선ID"]].turn_idx
    i = int(np.where((g["ars"].values == row["정류소ARS"]) & (g["seq"].values == row["순번"]))[0][0])
    end = turn if i < turn else len(g) - 1
    return list(g["ars"].values[i + 1:end + 1])

R["dars"] = [down_ars(r) for _, r in R.iterrows()]

keys = set(zip(R["정류소ARS"], R["노선"])) | {(a_, rt) for ars_list, rt in zip(R["dars"], R["노선"]) for a_ in ars_list}
acc, days_tot, nmon = {}, 0, 0
for p in sorted(glob.glob(D + "버스_정류장별시간대별승하차_2025월별/*.csv")):
    ym = re.search(r"(2025\d\d)", p).group(1); days_tot += calendar.monthrange(2025, int(ym[4:]))[1]; nmon += 1
    b = rd(p, dtype={"버스정류장ARS번호": str, "노선번호": str})
    b["ars"] = b["버스정류장ARS번호"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    b = b[[k in keys for k in zip(b["ars"], b["노선번호"])]]
    g = b.groupby(["ars", "노선번호"])[["22시승차총승객수", "23시승차총승객수", "00시승차총승객수", "00시하차총승객수", "1시승차총승객수", "23시하차총승객수", "22시하차총승객수"]].sum()
    for k, v in g.iterrows():
        a = acc.setdefault(k, np.zeros(7)); a += v.values
A = pd.DataFrame(acc, index=["b22", "b23", "b00", "a00", "b01", "a23", "a22"]).T
ALIGHT = {k: (v["a22"] + v["a23"] + v["a00"]) / days_tot for k, v in A.iterrows()}
A.index.names = ["정류소ARS", "노선"]
R = R.merge(A.reset_index(), on=["정류소ARS", "노선"], how="left").fillna({"b22": 0, "b23": 0, "b00": 0, "a00": 0, "b01": 0, "a23": 0, "a22": 0})
R["승차2223_일"] = (R["b22"] + R["b23"]) / days_tot
R["승차00_월"] = R["b00"] / nmon
R["승차00_일"] = R["b00"] / days_tot

tt = pd.read_csv(D + "지하철_열차운행시각표_20250930.csv", encoding="utf-8-sig", dtype=str)
tt = tt[(tt["WEEKTAG"] == "DAY") & tt["STATION_NM"].isin(STATIONS)].copy()
tt["arr"] = np.floor(tt["STT"].map(tm))
tt = tt.dropna(subset=["arr"])
last_dir = tt.groupby(["STATION_NM", "LINE", "INOUTTAG"])["arr"].max().reset_index()
st_last = last_dir.groupby("STATION_NM")["arr"].agg(last_min="min", last_max="max")
late = {s: np.sort(tt.loc[(tt["STATION_NM"] == s) & (tt["arr"] >= 23 * 60), "arr"].values) for s in STATIONS}
after0 = {s: v[v >= 24 * 60] for s, v in late.items()}

def judge(R, transfer=TRANSFER, col="model"):
    need = R["역"].map(st_last["last_max"]) + transfer
    gap = need - R[col]
    conn = [float(np.mean(after0[s] + transfer <= t)) if pd.notna(t) else np.nan for s, t in zip(R["역"], R[col])]
    return need, gap, conn

R["need"], R["gap"], R["연결률"] = judge(R)
R["판정"] = np.select([R["gap"] <= -10, R["gap"] <= 10, R["gap"] <= 30], ["연결", "경계(모델 오차 범위)", "30분 이내 부족"], "30분 초과 부족")
H = R[R["귀가방향"] & (R["하류정류소수"] > 0) & R["model"].notna()].copy().reset_index(drop=True)
H["dup"] = H.groupby(["역", "정류소ARS", "노선"])["노선"].transform("size")
H["w"] = H["승차2223_일"] / H["dup"]
H["id"] = list(zip(H["노선"], H["정류소ARS"], H["방향번호"]))

def down(row):
    g, turn = legs[row["노선ID"]].stops, legs[row["노선ID"]].turn_idx
    i = int(np.where((g["ars"].values == row["정류소ARS"]) & (g["seq"].values == row["순번"]))[0][0])
    end = turn if i < turn else len(g) - 1
    d = g.iloc[i + 1:end + 1]
    return d["lat"].values, d["lon"].values

H["dlat"], H["dlon"] = zip(*[down(r) for _, r in H.iterrows()])
H["rd"] = list(zip(H["역"], H["노선"], H["방향번호"]))
H["w_alight"] = [sum(ALIGHT.get((a_, rt), 0.0) for a_ in ars_list) for ars_list, rt in zip(H["dars"], H["노선"])]

def alight_points(g):
    recs = []
    for _, grp in g.sort_values("gap").groupby(["노선", "방향번호"], sort=False):
        pid = grp.iloc[0]["id"]
        seen, uniq = set(), []
        for _, r in grp.iterrows():
            for a_, la, lo in zip(r["dars"], r["dlat"], r["dlon"]):
                if a_ not in seen:
                    seen.add(a_); uniq.append((a_, la, lo))
        ws = [ALIGHT.get((a_, grp.iloc[0]["노선"]), 0.0) for a_, _, _ in uniq]
        if sum(ws) <= 0:
            ws = [grp["w"].sum() / len(uniq)] * len(uniq)
        for (a_, la, lo), w_ in zip(uniq, ws):
            if w_ > 0:
                recs.append({"id": pid, "lat": la, "lon": lo, "w": w_})
    return pd.DataFrame(recs)

night = {}
for rid, leg in legs.items():
    if not re.fullmatch(r"N\d\d", leg.route_name):
        continue
    g, turn = leg.stops, leg.turn_idx
    for s in STATIONS:
        for i in np.where(haversine_m(st_xy.loc[s, "위도"], st_xy.loc[s, "경도"], g["lat"].values, g["lon"].values) <= RADIUS_ST)[0]:
            end = turn if i < turn else len(g) - 1
            d = g.iloc[i + 1:end + 1]
            night.setdefault(s, []).append((leg.route_name, d["lat"].values, d["lon"].values))

def run_mclp(H, gapcol="gap", radius=RADIUS_COVER, stop_gain=STOP_GAIN, weight="board"):
    out = {}
    for s, g in H.groupby("역"):
        if weight == "alight":
            pts = alight_points(g)
        else:
            rows = [{"id": r["id"], "weight": r["w"], "lat": r["dlat"], "lon": r["dlon"]} for _, r in g.iterrows()]
            pts = build_demand_points(rows)
        base = {r["id"]: (r["dlat"], r["dlon"]) for _, r in g[g[gapcol] <= 0].iterrows()}
        base.update({f"{n}#{j}": (la, lo) for j, (n, la, lo) in enumerate(night.get(s, []))})
        cand = {r["id"]: (r["dlat"], r["dlon"]) for _, r in g[g[gapcol] > 0].iterrows()}
        res = greedy_max_cover(pts, base, cand, radius=radius, stop_gain=stop_gain, tie_break={r["id"]: r[gapcol] for _, r in g.iterrows()})
        pl, po = pts["lat"].values, pts["lon"].values
        bcov = np.zeros(len(pts), bool)
        for la, lo in base.values():
            bcov |= coverage(pl, po, la, lo, radius)
        pts = pts.assign(base=bcov, final=res["covered"])
        out[s] = {"res": res, "pts": pts, "share": covered_share_by_row(pts, res["covered"]), "n_base": int((g[gapcol] <= 0).sum()), "night": sorted({n for n, _, _ in night.get(s, [])}),
                  "base": base, "cand": cand}
    return out

def exact_best(pts, base, cand, radius=RADIUS_COVER, kmax=6):
    from itertools import combinations
    pl, po, w = pts["lat"].values, pts["lon"].values, pts["w"].values
    W = w.sum()
    bcov = np.zeros(len(pts), bool)
    for la, lo in base.values():
        bcov |= coverage(pl, po, la, lo, radius)
    ids = list(cand); masks = [coverage(pl, po, *cand[i], radius) for i in ids]
    out = {0: (w[bcov].sum() / W, [])}
    for k in range(1, min(kmax, len(ids)) + 1):
        best, bset = -1, None
        for comb in combinations(range(len(ids)), k):
            cov = bcov.copy()
            for j in comb:
                cov |= masks[j]
            v = w[cov].sum() / W
            if v > best:
                best, bset = v, [ids[j] for j in comb]
        out[k] = (best, bset)
    return out

def cov_fixed(pts, base, cand, chosen, radius=RADIUS_COVER):
    pl, po, w = pts["lat"].values, pts["lon"].values, pts["w"].values
    cov = np.zeros(len(pts), bool)
    for la, lo in base.values():
        cov |= coverage(pl, po, la, lo, radius)
    for i in chosen:
        if i in cand:
            cov |= coverage(pl, po, *cand[i], radius)
    return float(w[cov].sum() / w.sum())

M = run_mclp(H)
MF = run_mclp(H, stop_gain=0.0)
MA = run_mclp(H, weight="alight")
EX = {s: exact_best(M[s]["pts"], M[s]["base"], M[s]["cand"], kmax=6) for s in M}
EXA = {s: exact_best(MA[s]["pts"], MA[s]["base"], MA[s]["cand"], kmax=6) for s in MA}
FIX = {}
for s in STATIONS:
    sb = [i[0] for i in M[s]["res"]["selected"]]; sa = [i[0] for i in MA[s]["res"]["selected"]]
    FIX[s] = {"sel_b": sb, "sel_a": sa, "k": len(sb),
              "b_on_b": M[s]["res"]["final_cov"], "b_on_a": cov_fixed(MA[s]["pts"], MA[s]["base"], MA[s]["cand"], sb),
              "a_on_a": MA[s]["res"]["final_cov"], "a_on_b": cov_fixed(M[s]["pts"], M[s]["base"], M[s]["cand"], sa),
              "a_best_k": EXA[s][min(len(sb), max(EXA[s]))][0], "b_best_k": EX[s][min(len(sb), max(EX[s]))][0],
              "base_a": MA[s]["res"]["base_cov"], "base_b": M[s]["res"]["base_cov"]}

def waits(s, trips):
    arr = after0[s] + TRANSFER
    out = []
    for a in arr:
        later = [t for t in trips if t >= a]
        out.append(min(later) - a if later else np.nan)
    return np.array(out)

WAIT = {}
for s in STATIONS:
    need = st_last.loc[s, "last_max"] + TRANSFER
    w1 = waits(s, [need])
    best = None
    for t1 in np.arange(24 * 60, need, 1.0):
        w2 = waits(s, [t1, need])
        if best is None or np.nanmax(w2) < best[1]:
            best = (t1, float(np.nanmax(w2)), float(np.nanmean(w2)), float(np.mean(w2 <= 30)))
    WAIT[s] = {"need": need, "one_mean": float(np.nanmean(w1)), "one_max": float(np.nanmax(w1)), "one_le30": float(np.mean(w1 <= 30)),
               "two_t1": best[0], "two_max": best[1], "two_mean": best[2], "two_le30": best[3], "n_train": len(after0[s])}
full_order = {s: o["res"]["selected"] for s, o in MF.items()}
sel = {}
for s, o in M.items():
    for k, (i, gain, cum) in enumerate(o["res"]["selected"]):
        sel[(s, i)] = (k + 1, gain, cum)
H["선택순서"] = [sel.get((s, i), (None,))[0] for s, i in zip(H["역"], H["id"])]
H["커버증가"] = [sel.get((s, i), (None, np.nan))[1] for s, i in zip(H["역"], H["id"])]
H["행커버"] = [M[s]["share"].get(i, np.nan) for s, i in zip(H["역"], H["id"])]
H["share_dir"] = H["b23"] / H.groupby("역")["b23"].transform("sum")
H["필요연장"] = [int(math.ceil(g / 5) * 5) if g > 0 else 0 for g in H["gap"]]
for x in (15, 30, 45, 60):
    H[f"연결률+{x}"] = [float(np.mean(after0[s] + TRANSFER <= t + x)) for s, t in zip(H["역"], H["model"])]

def action(r):
    if r["gap"] <= 0:
        return "연결됨"
    if pd.notna(r["선택순서"]):
        return f"기점 막차 {r['필요연장']}분 연장" if r["필요연장"] <= 15 else f"지하철 막차 연계 1회 추가(기점 +{r['필요연장']}분)"
    if r["행커버"] >= 0.8:
        return "선택 노선·올빼미로 대체 연결"
    return "일부 대체 + 심야 DRT 검토" if r["행커버"] > 0 else "심야 DRT 검토"

H["조치"] = H.apply(action, axis=1)

team_prop = [("사당", "5530", "20253", 30), ("사당", "5524", "20253", 25), ("강변", "강동01", None, 30), ("강변", "광진03", None, 30), ("강변", "광진05", "05585", 30)]
TP = []
for s, r, ars, x in team_prop:
    g = H[(H["역"] == s) & (H["노선"] == r)]
    if ars:
        g = g[g["정류소ARS"] == ars]
    row = g.sort_values("w", ascending=False).iloc[0]
    TP.append({"역": s, "노선": r, "정류소명": row["정류소명"], "방면": row["방면"], "현재(추정)": hhmm(row["model"]), "팀 제안 연장": x, "개선 후": hhmm(row["model"] + x),
               "연결률 전": row["연결률"], "연결률 후": float(np.mean(after0[s] + TRANSFER <= row["model"] + x)), "전 열차 연결 필요 연장": row["필요연장"],
               "API 예정": hhmm(row["api"])})
TP = pd.DataFrame(TP)

S = []
for tr in (5, 6, 7):
    need, gap, conn = judge(H, transfer=tr)
    S.append({"구분": f"환승 {tr}분", "연결(부족≤0)": int((gap <= 0).sum()), "부족 중앙값": float(gap.median()), "15분 이내": int(((gap > 0) & (gap <= 15)).sum()), "30분 초과": int((gap > 30).sum())})
for sp in (0.70, 0.85, 1.00):
    H2 = H.copy(); H2["m2"] = H2.apply(lambda r: model_time(r, sp), axis=1)
    need, gap, conn = judge(H2, col="m2"); H2["gap2"] = gap
    m2 = run_mclp(H2, gapcol="gap2")
    S.append({"구분": f"속도계수 {sp:.2f}", "연결(부족≤0)": int((gap <= 0).sum()), "부족 중앙값": float(gap.median()), "15분 이내": int(((gap > 0) & (gap <= 15)).sum()), "30분 초과": int((gap > 30).sum()),
              "선택 노선": {s: len(o["res"]["selected"]) for s, o in m2.items()}, "커버": {s: (round(o["res"]["base_cov"], 3), round(o["res"]["final_cov"], 3)) for s, o in m2.items()}})
for rad in (300, 500):
    m3 = run_mclp(H, radius=rad)
    S.append({"구분": f"커버 반경 {rad}m", "선택 노선": {s: len(o["res"]["selected"]) for s, o in m3.items()}, "커버": {s: (round(o["res"]["base_cov"], 3), round(o["res"]["final_cov"], 3)) for s, o in m3.items()}})
S = pd.DataFrame(S)

cols = ["역", "노선", "유형", "정류소명", "정류소ARS", "방향구간", "방면", "순번", "총정류소수", "기점막차", "운행시간", "승차2223_일", "승차00_월", "판정", "조치", "선택순서", "필요연장"]
X = H[cols].copy()
X["추정 막차"] = H["model"].map(hhmm); X["필요 출발"] = H["need"].map(hhmm); X["부족(분)"] = H["gap"].round(0)
X["자정 이후 열차 연결률"] = H["연결률"].round(3); X["API 예정(검증용)"] = H["api"].map(hhmm)
with pd.ExcelWriter("out/final/최종_사당강변_막차연계_처방표.xlsx") as w:
    X.sort_values(["역", "선택순서", "부족(분)"]).to_excel(w, sheet_name="귀가방향 처방표", index=False)
    TP.to_excel(w, sheet_name="팀 제안안 검증", index=False)
    S.astype(str).to_excel(w, sheet_name="민감도", index=False)
    last_dir.assign(막차도착=last_dir["arr"].map(hhmm)).drop(columns="arr").to_excel(w, sheet_name="지하철 평일 막차", index=False)
U = H.sort_values("gap").groupby("rd").first().reset_index()
pickle.dump({"R": R, "H": H, "U": U, "M": {s: {"res": o["res"], "pts": o["pts"], "n_base": o["n_base"], "night": o["night"]} for s, o in M.items()},
             "MA": {s: {"res": o["res"], "pts": o["pts"]} for s, o in MA.items()}, "EX": EX, "EXA": EXA, "WAIT": WAIT, "FIX": FIX, "TP": TP, "S": S,
             "full_order": full_order, "night": night, "st_last": st_last, "last_dir": last_dir, "late": late, "after0": after0, "st_xy": st_xy, "days": days_tot, "nmon": nmon}, open("out/final/_final.pkl", "wb"))

v = R.dropna(subset=["api", "model"])
print(f"[4-1] 행 {len(R)} (사당 {(R['역']=='사당').sum()}, 강변 {(R['역']=='강변').sum()}) | 귀가방향 분석 대상 {len(H)}행")
print(f"[고유 노선-방향] {len(U)}개 (정류소 포함 {len(H)}행) | 연결 {(U['gap'] <= 0).sum()} | 중앙값 {U['gap'].median():.1f}")
for s in STATIONS:
    print(f"[하차 가중 {s}] 커버 {MA[s]['res']['base_cov']:.3f} → {MA[s]['res']['final_cov']:.3f}, 선택 {[i[0][0] for i in MA[s]['res']['selected']]} | 수요점 {len(MA[s]['pts'])}개(중복 제거)")
    print(f"[고정 민감도 {s}] 채택안 {[i[0] for i in FIX[s]['sel_b']]}: 승차 {FIX[s]['b_on_b']:.3f} / 하차 {FIX[s]['b_on_a']:.3f} | 하차 최적안 {[i[0] for i in FIX[s]['sel_a']]}: 하차 {FIX[s]['a_on_a']:.3f} / 승차 {FIX[s]['a_on_b']:.3f} | 하차 k={FIX[s]['k']} 전수최대 {FIX[s]['a_best_k']:.3f}")
    print(f"[전수 비교 {s}] 승차가중: " + ", ".join(f"k={k}:{v[0]:.3f}" for k, v in EX[s].items()) + " | 탐욕: " + ", ".join(f"{k}:{c:.3f}" for k, c in M[s]["res"]["curve"][:7]))
    print(f"[대기 {s}] 1회({hhmm(WAIT[s]['need'])}): 평균 {WAIT[s]['one_mean']:.0f}분, 최대 {WAIT[s]['one_max']:.0f}분, 30분 이내 {WAIT[s]['one_le30']:.0%} | 2회({hhmm(WAIT[s]['two_t1'])}+{hhmm(WAIT[s]['need'])}): 최대 {WAIT[s]['two_max']:.0f}분, 평균 {WAIT[s]['two_mean']:.0f}분, 30분 이내 {WAIT[s]['two_le30']:.0%}")
print(f"[재현] 팀 산출 막차와 1분 이내 일치 {int((R['repro_diff'] <= 1).sum())}/{int(R['team_model'].notna().sum())}")
print(f"[검증] API 대조 {len(v)}행: 평균 {v['err'].mean():+.1f}분, 중앙값 {v['err'].median():+.1f}, MAE {v['err'].abs().mean():.1f}, ±10분 {(v['err'].abs()<=10).mean():.1%}, ±20분 {(v['err'].abs()<=20).mean():.1%}")
print("[지하철 평일 막차]", {f"{a} {b}호선 {c}": hhmm(d) for a, b, c, d in last_dir.values})
print("[자정 이후 도착 열차 수]", {s: len(v2) for s, v2 in after0.items()}, "| 23시 이후", {s: len(v2) for s, v2 in late.items()})
print("[판정]", H.groupby(["역", "판정"]).size().unstack(fill_value=0).to_dict("index"))
print("[부족 중앙값]", H.groupby("역")["gap"].median().round(1).to_dict(), "| 연결률 중앙값", H.groupby("역")["연결률"].median().round(3).to_dict())
for s, o in M.items():
    r = o["res"]
    print(f"[MCLP {s}] 기존 연결 {o['n_base']}개·올빼미 {o['night']} | 커버 {r['base_cov']:.3f} → {r['final_cov']:.3f} | 선택 {len(r['selected'])}개")
print(H[H["선택순서"].notna()].sort_values(["역", "선택순서"])[["역", "선택순서", "노선", "유형", "정류소명", "방면", "기점막차", "gap", "필요연장", "연결률", "연결률+30", "연결률+60", "커버증가", "승차2223_일", "승차00_월"]].round(2).assign(추정=H["model"].map(hhmm), api=H["api"].map(hhmm)).to_string(index=False))
print("[조치 분포]", H.groupby(["역", H["조치"].str.replace(r"\(.*\)|\d+분 ", "", regex=True)]).size().to_dict())
print(TP.round(3).to_string(index=False))
print(S.to_string(index=False))
