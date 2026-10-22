import os, re, json, time, math, datetime, urllib.request, urllib.parse
import pandas as pd, numpy as np

KEY = os.environ.get("SEOUL_BUS_KEY", "").strip()
NOW = datetime.datetime.now()
STAMP = NOW.strftime("%Y-%m-%d %H:%M")
D = "data/"
CACHE = "api/cache/"
os.makedirs(CACHE, exist_ok=True)
TOP = ["사당", "강변", "창동", "대림", "영등포구청", "신대방", "성수", "건대입구"]
BENCH = 34.0
IN_STATION = 3.0
WALK_MPM = 83.3
DETOUR = 1.3

def rd(p, **kw):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(p, encoding=enc, **kw)
        except UnicodeDecodeError:
            pass

def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))

def norm(n):
    n = re.sub(r"\(.*?\)", "", str(n)).strip()
    return n[:-1] if len(n) > 2 and n.endswith("역") else n

def tmin(s):
    if pd.isna(s) or s == "00:00:00":
        return np.nan
    h, m, sec = map(int, s.split(":"))
    if h < 4:
        h += 24
    return h * 60 + m + sec / 60

def api_min(v):
    v = re.sub(r"\D", "", str(v or ""))
    if len(v) == 14:
        v = v[8:]
    if len(v) < 4:
        return np.nan
    h, m = int(v[:2]), int(v[2:4])
    if h < 4:
        h += 24
    return h * 60 + m + (int(v[4:6]) / 60 if len(v) >= 6 else 0)

def hhmm(x):
    return "" if pd.isna(x) else f"{int(x // 60) % 24:02d}:{int(x % 60):02d}"

def call_bustime(ars, rid):
    fn = f"{CACHE}bustime_{ars}_{rid}.json"
    if os.path.exists(fn):
        return json.load(open(fn, encoding="utf-8"))
    if not KEY:
        return {"ok": False, "msg": "키 없음·캐시 없음"}
    url = "http://ws.bus.go.kr/api/rest/stationinfo/getBustimeByStation?" + urllib.parse.urlencode(
        {"serviceKey": KEY, "arsId": ars, "busRouteId": rid, "resultType": "json"})
    res = {"ok": False, "msg": "", "item": {}, "queried_at": STAMP}
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                j = json.loads(r.read().decode("utf-8"))
            hdr = j.get("msgHeader", {}) or {}
            items = (j.get("msgBody", {}) or {}).get("itemList") or []
            res.update(msg=hdr.get("headerMsg", ""), ok=bool(items) and str(hdr.get("headerCd")) == "0", item=items[0] if items else {})
            break
        except Exception as e:
            res["msg"] = f"EXC:{e}"
            time.sleep(1.5)
    json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
    time.sleep(0.12)
    return res

c18 = rd(D + "지하철_역좌표_1-8호선_20250814.csv").rename(columns={"고유역번호(외부역코드)": "code"})
c9 = rd(D + "지하철_역좌표_9호선2-3단계_20260131.csv").rename(columns={"역번호": "code"})
coords = pd.concat([c18[["호선", "code", "역명", "위도", "경도"]], c9[["호선", "code", "역명", "위도", "경도"]]], ignore_index=True)
coords["code"] = [f"{int(float(v)):04d}" for v in coords["code"]]
coords["name"] = [norm(v) for v in coords["역명"]]
code2name = dict(zip(coords.code, coords.name))
st_xy = coords.groupby("name")[["위도", "경도"]].mean()

t = rd(D + "지하철_열차운행시각표_20260902.csv", dtype=str)
t = t[t["주중주말"] == "DAY"].copy()
t["arr"] = t["열차도착시간"].map(tmin)
t = t.dropna(subset=["arr"])
t["name"] = t["역사코드"].map(code2name)
t = t[t.name.isin(TOP)]
lastdir = t.groupby(["name", "호선", "방향"])["arr"].max().reset_index()
st_last = lastdir.groupby("name")["arr"].agg(last_min="min", last_max="max")
st_last_txt = lastdir.assign(txt=lambda z: z["호선"] + "호선 " + z["방향"] + " " + z["arr"].map(hhmm)).groupby("name")["txt"].apply(" / ".join)

stops = pd.read_excel(D + "버스_정류소위치_20260902.xlsx", dtype={"ARS_ID": str})
stops["ars"] = stops["ARS_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
stop_xy = stops.drop_duplicates("ars").set_index("ars")[["Y좌표", "X좌표", "정류소명"]]

sq = pd.read_excel(D + "버스_노선별경유정류소순서_20251209.xlsx", dtype={"ARS_ID": str, "노선명": str})
sq["ars"] = sq["ARS_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
sq = sq.sort_values(["ROUTE_ID", "순번"])
rid_by_name = pd.read_excel(D + "버스_노선ID목록_20251209.xlsx", dtype={"노선명": str}).drop_duplicates("노선명").set_index("노선명")["ROUTEID"].to_dict()
for nm, rid in sq.drop_duplicates("노선명").set_index("노선명")["ROUTE_ID"].items():
    rid_by_name.setdefault(nm, rid)

legs = {}
for rid, g in sq.groupby("ROUTE_ID"):
    g = g.reset_index(drop=True)
    dd = hav(g["Y좌표"].iloc[0], g["X좌표"].iloc[0], g["Y좌표"].values, g["X좌표"].values)
    turn = int(np.argmax(dd))
    legs[rid] = (g, turn)

def directions(rid, ars):
    if rid not in legs:
        return []
    g, turn = legs[rid]
    out = []
    for i in np.where(g["ars"].values == ars)[0]:
        if i <= turn:
            end = turn; dest = g["정류소명"].iloc[turn]
        else:
            end = len(g) - 1; dest = g["정류소명"].iloc[-1]
        ds = g.iloc[i + 1:end + 1]
        out.append({"순번": int(g["순번"].iloc[i]), "방면": dest, "down_lat": ds["Y좌표"].values, "down_lon": ds["X좌표"].values, "n_down": len(ds)})
    return out

night = {}
for nm in sq[sq["노선명"].str.startswith("N")]["노선명"].unique():
    rid = rid_by_name.get(nm)
    g, turn = legs[rid]
    for stn in TOP:
        la, lo = st_xy.loc[stn, "위도"], st_xy.loc[stn, "경도"]
        near = np.where(hav(la, lo, g["Y좌표"].values, g["X좌표"].values) <= 300)[0]
        for i in near:
            end = turn if i <= turn else len(g) - 1
            ds = g.iloc[i + 1:end + 1]
            night.setdefault(stn, []).append((nm, ds["Y좌표"].values, ds["X좌표"].values))

def night_cover(stn, dlat, dlon):
    best, who = 0.0, ""
    if len(dlat) == 0:
        return 0.0, ""
    for nm, nla, nlo in night.get(stn, []):
        if len(nla) == 0:
            continue
        cov = np.mean([hav(a, b, nla, nlo).min() <= 400 for a, b in zip(dlat, dlon)])
        if cov > best:
            best, who = cov, nm
    return best, who

hp = pd.read_csv("out/policy_home_pairs_2025.csv", dtype={"노선번호": str})
mix = pd.read_csv("out/policy_station_mix_2025.csv")[["name", "alight_24p"]]
hp = hp.merge(hp.groupby("name").b23.sum().rename("st_b23"), on="name").merge(mix, on="name", how="left")
hp["link"] = hp.alight_24p * hp.b23 / hp.st_b23
route_link = hp.groupby("노선번호").link.sum()

b26 = rd(D + "버스_정류장별시간대별승하차_202608.csv", dtype={"버스정류장ARS번호": str, "노선번호": str})
b26["ars"] = b26["버스정류장ARS번호"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
b26["h00"] = b26["00시승차총승객수"] + b26["00시하차총승객수"]
b26["h01"] = b26["1시승차총승객수"] + b26["1시하차총승객수"]
act26 = b26.groupby(["ars", "노선번호"])[["h00", "h01"]].sum()

team = pd.read_excel(D + "분석_2025Q4_심야교통취약지역_역접근성_API조회대상포함.xlsx", sheet_name="API조회대상", header=5)
team = team.dropna(subset=["ARS_ID", "busRouteId"])
trow = []
for _, r in team.iterrows():
    ars = f"{int(r.ARS_ID):05d}"; rid = int(r.busRouteId)
    res = call_bustime(ars, rid); it = res.get("item", {})
    trow.append({"검토순위": r["검토순위"], "지역": r["지역"], "정류소명": r["정류소명"], "ARS_ID": ars, "정류소ID(stId)": it.get("stId", ""),
                 "노선명": r["노선명"], "busRouteId": rid, "첫차": hhmm(api_min(it.get("firstBusTm"))), "막차": hhmm(api_min(it.get("lastBusTm"))),
                 "막차_원값": it.get("lastBusTm", ""), "조회일시": res.get("queried_at", STAMP), "조회성공": "Y" if res.get("ok") else "N", "응답메시지": res.get("msg", "")})
team_df = pd.DataFrame(trow)
os.makedirs("api", exist_ok=True)
team_df.to_csv(f"api/api_lastbus_팀94조합_{NOW:%Y%m%d}.csv", index=False, encoding="utf-8-sig")
print(f"[팀 94조합] {len(team_df)}건 | 성공 {(team_df.조회성공 == 'Y').sum()} | 실패 {(team_df.조회성공 == 'N').sum()}")

T = pd.read_excel("out/8개역_귀가방향노선_2025월별판정.xlsx", dtype={"정류소ARS": str, "노선": str})
rows = []
for _, r in T.iterrows():
    stn, route, ars = r["역"], r["노선"], r["정류소ARS"].zfill(5)
    rid = rid_by_name.get(route)
    res = call_bustime(ars, rid) if rid else {"ok": False, "msg": "노선ID 없음", "item": {}}
    it = res.get("item", {})
    last_bus = api_min(it.get("lastBusTm"))
    la, lo = st_xy.loc[stn, "위도"], st_xy.loc[stn, "경도"]
    if ars in stop_xy.index:
        dist = float(hav(la, lo, stop_xy.loc[ars, "Y좌표"], stop_xy.loc[ars, "X좌표"])); sname = stop_xy.loc[ars, "정류소명"]
    else:
        dist, sname = np.nan, ""
    transfer = IN_STATION + dist * DETOUR / WALK_MPM
    dirs = directions(rid, ars) or [{"순번": None, "방면": "미상", "down_lat": np.array([]), "down_lon": np.array([]), "n_down": 0}]
    for dv in dirs:
        cov, nroute = night_cover(stn, dv["down_lat"], dv["down_lon"])
        need = st_last.loc[stn, "last_max"] + transfer
        gap = need - last_bus if res.get("ok") and not np.isnan(last_bus) else np.nan
        part = (last_bus >= st_last.loc[stn, "last_min"] + transfer) if not np.isnan(gap) else False
        rl = float(route_link.get(route, 0.0))
        if np.isnan(gap):
            rx = "판정 보류(API 미확인)"
        elif gap <= 0:
            rx = "연결됨"
        elif gap <= 15:
            rx = f"막차 {int(math.ceil(gap / 5) * 5)}분 연장"
        elif rl >= BENCH:
            rx = "막차 연계 1회 추가"
        elif cov >= 0.5:
            rx = f"올빼미 {nroute} 연계"
        else:
            rx = "심야 DRT 검토"
        a26 = act26.loc[(ars, route)] if (ars, route) in act26.index else None
        flag = ""
        if res.get("ok") and not np.isnan(last_bus) and a26 is not None:
            if last_bus >= 24 * 60 + 10 and a26.h00 == 0:
                flag = "예정 막차 00시대인데 2026-08 00시 승하차 0"
            elif last_bus < 24 * 60 and a26.h00 > 0:
                flag = "예정 막차 자정 전인데 2026-08 00시 승하차 있음"
        rows.append({"역": stn, "노선": route, "유형": r["유형"], "정류소ARS": ars, "정류소명": sname, "busRouteId": rid, "순번": dv["순번"],
                     "방면": dv["방면"], "API 막차": hhmm(last_bus), "막차_분": last_bus, "API 조회": "Y" if res.get("ok") else "N",
                     "역-정류소 직선거리(m)": round(dist), "환승 소요(분)": round(transfer, 1), "지하철 막차 최종": hhmm(st_last.loc[stn, "last_max"]),
                     "지하철 막차 최초": hhmm(st_last.loc[stn, "last_min"]), "필요 출발 시각": hhmm(need), "부족(분)": None if np.isnan(gap) else round(gap, 1),
                     "일부 방향 연결": "Y" if part else "", "2025 연간 판정": r["연간 대표 판정"], "22~23시 승차(일평균)": r["22~23시 승차(일평균)"],
                     "노선 연계수요(명/평일)": round(rl, 1), "올빼미 방향 일치율": round(cov, 2), "올빼미 노선": nroute, "처방": rx, "실측 대조": flag,
                     "down_lat": dv["down_lat"], "down_lon": dv["down_lon"]})
P = pd.DataFrame(rows)
P.to_csv(f"api/api_lastbus_8역_{NOW:%Y%m%d}.csv", index=False, encoding="utf-8-sig",
         columns=["역", "노선", "정류소ARS", "busRouteId", "API 막차", "API 조회", "순번", "방면"])
with pd.ExcelWriter("out/2단계_8개역_막차연계_처방표.xlsx") as w:
    P.drop(columns=["막차_분", "down_lat", "down_lon"]).to_excel(w, sheet_name="처방표", index=False)
    team_df.to_excel(w, sheet_name="팀94조합_API", index=False)
    pd.DataFrame({"역": st_last_txt.index, "방향별 지하철 막차(평일, 2026-09-02 시각표)": st_last_txt.values}).to_excel(w, sheet_name="지하철막차", index=False)
P.to_pickle("out/_stage2.pkl")
import pickle; pickle.dump({"night": night, "st_last": st_last, "st_last_txt": st_last_txt}, open("out/_stage2_aux.pkl", "wb"))

print(f"[8개 역] 노선-정류소-방향 {len(P)}행 | API 성공 {(P['API 조회'] == 'Y').sum()} | 실패 {(P['API 조회'] == 'N').sum()}")
print("처방 분포:", P["처방"].str.replace(r"올빼미 N\d+ 연계", "올빼미 연계", regex=True).str.replace(r"막차 \d+분 연장", "막차 5~15분 연장", regex=True).value_counts().to_dict())
print("실측 대조 플래그:", P["실측 대조"].value_counts().to_dict())
for tr in (5, 10):
    g = P.dropna(subset=["부족(분)"]).copy()
    g["gap_t"] = g["막차_분"].rsub(g["역"].map(st_last.last_max) + tr)
    print(f"민감도 환승 {tr}분 고정: 연결 {(g.gap_t <= 0).sum()} / 15분 이내 {((g.gap_t > 0) & (g.gap_t <= 15)).sum()} / 15분 초과 {(g.gap_t > 15).sum()}")
print(P.groupby("역").apply(lambda g: pd.Series({"행": len(g), "연결됨": (g.처방 == "연결됨").sum(), "부족 중앙값": g["부족(분)"].median()}), include_groups=False))
