import pandas as pd, numpy as np, re, os
os.makedirs("out", exist_ok=True)
D = "data/"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
TRANSFER = 5
DAYS = 31

def rd(p, **kw):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(p, encoding=enc, **kw)
        except UnicodeDecodeError:
            pass

def tmin(s):
    if pd.isna(s) or s == "00:00:00":
        return np.nan
    h, m, sec = map(int, s.split(":"))
    if h < 4:
        h += 24
    return h * 60 + m + sec / 60

def norm(n):
    n = re.sub(r"\(.*?\)", "", str(n)).strip()
    return n[:-1] if len(n) > 2 and n.endswith("역") else n

def code4(v):
    return f"{int(float(v)):04d}"

def hhmm(x):
    return "" if pd.isna(x) else f"{int(x//60):02d}:{int(x%60):02d}"

c18 = rd(D + "지하철_역좌표_1-8호선_20250814.csv").rename(columns={"고유역번호(외부역코드)": "code"})
c9 = rd(D + "지하철_역좌표_9호선2-3단계_20260131.csv").rename(columns={"역번호": "code"})
coords = pd.concat([c18[["호선", "code", "역명", "위도", "경도"]], c9[["호선", "code", "역명", "위도", "경도"]]], ignore_index=True)
coords["code"] = [code4(v) for v in coords["code"]]
coords["name"] = [norm(v) for v in coords["역명"]]
coords["호선"] = coords["호선"].astype(str)
code2name = dict(zip(coords.code, coords.name))
print("좌표 역-호선:", len(coords), "| 고유 역명:", coords.name.nunique())

t = rd(D + "지하철_열차운행시각표_20260902.csv", dtype=str)
t = t[t["주중주말"] == "DAY"].copy()
t["arr"] = t["열차도착시간"].map(tmin)
t = t.dropna(subset=["arr"])
t["name_tt"] = t["역사명"].map(norm)
t18 = t[t["호선"] != "9"].copy(); t18["name"] = t18["역사코드"].map(code2name)
names9 = set(coords.loc[coords["호선"] == "9", "name"])
t9 = t[(t["호선"] == "9") & (t["name_tt"].isin(names9))].copy(); t9["name"] = t9["name_tt"]
tt = pd.concat([t18.dropna(subset=["name"]), t9])
last_dir = tt.groupby(["name", "호선", "방향"])["arr"].max().reset_index()
st_last = last_dir.groupby("name")["arr"].agg(last_min="min", last_max="max").reset_index()
print("막차 산출 역명:", st_last.shape[0], "/ 좌표 역명", coords.name.nunique())

d = rd(D + "지하철_역별일별시간대별승하차_2025.csv")
d["date"] = pd.to_datetime(d["수송일자"])
hol = pd.to_datetime(["2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-03",
                      "2025-05-05", "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03",
                      "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-12-25"])
wk = d[(d.date.dt.weekday < 5) & (~d.date.isin(hol)) & (d["승하차구분"] == "하차")].copy()
ndays = wk.date.nunique()
wk["name"] = [code2name.get(code4(v)) for v in wk["역번호"]]
print("평일 일수:", ndays, "| 이름 미매칭 역:", wk.loc[wk.name.isna(), "역명"].unique()[:10].tolist())
dem = wk.groupby("name")[["22-23시간대", "23-24시간대", "24시이후"]].sum() / ndays
dem.columns = ["alight_22", "alight_23", "alight_24p"]
dem = dem.sort_values("alight_24p", ascending=False).reset_index()
dem["rank"] = dem.index + 1
tot24 = dem.alight_24p.sum()
print(f"평일 일평균 24시이후 하차 합계({len(dem)}개 역명): {tot24:,.0f}명 | 23-24시: {dem.alight_23.sum():,.0f}명")
for k in (10, 30, 50):
    print(f"  상위 {k}역 24시이후 점유: {dem.alight_24p.head(k).sum()/tot24:.1%}")

x = pd.read_excel(D + "버스_정류소위치_20260902.xlsx", dtype={"ARS_ID": str})
x = x[x["정류소타입"] != "한강선착장"].copy()
x["ars"] = x["ARS_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
st_xy = coords.groupby("name")[["위도", "경도"]].mean().reset_index()
lat1 = np.radians(st_xy["위도"].values)[:, None]; lon1 = np.radians(st_xy["경도"].values)[:, None]
lat2 = np.radians(x["Y좌표"].values)[None, :]; lon2 = np.radians(x["X좌표"].values)[None, :]
a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
dist = 2 * 6371000 * np.arcsin(np.sqrt(a))
ii, jj = np.where(dist <= 300)
near = pd.DataFrame({"name": st_xy["name"].values[ii], "ars": x["ars"].values[jj], "stop_nm": x["정류소명"].values[jj],
                     "dist_m": dist[ii, jj].round(0)})

b = rd(D + "버스_정류장별시간대별승하차_202608.csv", dtype={"버스정류장ARS번호": str, "교통수단타입코드": str})
b = b[b["교통수단타입명"].isin(["서울간선버스", "서울지선버스", "서울마을버스", "서울순환버스"])].copy()
b["ars"] = b["버스정류장ARS번호"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
b["b23"] = b["23시승차총승객수"]; b["a23"] = b["23시하차총승객수"]
b["b00"] = b["00시승차총승객수"]; b["a00"] = b["00시하차총승객수"]
b["b22"] = b["22시승차총승객수"]
b["type"] = b["교통수단타입명"].str.replace("서울", "").str.replace("버스", "")
nb = near.merge(b[["ars", "노선번호", "노선명", "type", "b22", "b23", "a23", "b00", "a00"]], on="ars", how="inner")
nb = nb.merge(st_last, on="name", how="left")
home = nb[nb.b23 >= DAYS].copy()
home["ends_before_00"] = (home.b00 + home.a00) == 0
home["strong_gap"] = home.ends_before_00 & (home.last_min >= 24 * 60 + TRANSFER)
home["min_ext_lb"] = np.where(home.strong_gap, home.last_min + TRANSFER - 24 * 60, np.nan)
home.to_csv("out/home_route_stops.csv", index=False, encoding="utf-8-sig")
print(f"\n귀가방향 노선-정류장(역 300m, 23시 승차 월31+): {len(home)}쌍 | 00시 이전 종료: {home.ends_before_00.sum()} ({home.ends_before_00.mean():.1%})")
print(home.groupby("type").agg(pairs=("ars", "size"), ends_before_00=("ends_before_00", "sum"), strong=("strong_gap", "sum")).assign(share=lambda z: (z.ends_before_00/z.pairs).round(3)))

print("\n역의 가장 이른 방향 막차 도착(평일) 분포:")
print(pd.cut(st_last.last_min, [0, 24*60, 24*60+TRANSFER, 24*60+15, 24*60+30, 24*60+45, 25*60, 26*60], right=False,
             labels=["~24:00", "24:00~05", "00:05~15", "00:15~30", "00:30~45", "00:45~01:00", "01:00~"]).value_counts().sort_index())
print("\n강한 단절 노선-정류장의 필요 연장 하한(분) 분포:")
lb = home.loc[home.strong_gap, "min_ext_lb"]
print(pd.cut(lb, [0, 5, 15, 30, 45, 60, 120], right=True).value_counts().sort_index())
print(f"  하한 15분 이하 비율: {(lb<=15).mean():.1%} | 중앙값 {lb.median():.0f}분")

def agg(g):
    s = g[g.strong_gap]
    return pd.Series({
        "n_home_routes": g["노선번호"].nunique(),
        "n_gap_routes": s["노선번호"].nunique(),
        "n_gap_maeul": s.loc[s.type == "마을", "노선번호"].nunique(),
        "b23_home": g.b23.sum(), "b23_gap": s.b23.sum(),
        "gap_routes": ", ".join(sorted(s["노선번호"].astype(str).unique())[:10]),
    })
summ = home.groupby("name")[["노선번호", "type", "b23", "strong_gap"]].apply(agg).reset_index()
res = dem.merge(st_last, on="name", how="left").merge(summ, on="name", how="left")
res["gap_share_b23"] = (res.b23_gap / res.b23_home).fillna(0)
res["exposure_proxy"] = res.alight_24p * res.gap_share_b23
res["last_min_hhmm"] = res.last_min.map(hhmm); res["last_max_hhmm"] = res.last_max.map(hhmm)
res.to_csv("out/station_summary_v2.csv", index=False, encoding="utf-8-sig")
for k in (30, 50, len(res)):
    sub = res.head(k)
    print(f"[24시이후 하차 상위 {k}역] 단절노선 보유 역 {(sub.n_gap_routes>0).sum()}/{k} | 단절 노선 {int(sub.n_gap_routes.sum())} (마을 {int(sub.n_gap_maeul.sum())})"
          f" | 23시 귀가승차 중 단절노선 비중 {sub.b23_gap.sum()/sub.b23_home.sum():.1%} | 노출 대리지표 {sub.exposure_proxy.sum():,.0f}명/평일")
cols = ["rank", "name", "alight_23", "alight_24p", "last_min_hhmm", "last_max_hhmm", "n_home_routes", "n_gap_routes", "n_gap_maeul", "gap_share_b23", "exposure_proxy", "gap_routes"]
fmt = {"alight_23": 0, "alight_24p": 0, "gap_share_b23": 2, "exposure_proxy": 0}
print("\n24시이후 하차 상위 30역:"); print(res[cols].head(30).round(fmt).to_string(index=False))
print("\n노출 대리지표 상위 15역:"); print(res.sort_values("exposure_proxy", ascending=False)[cols].head(15).round(fmt).to_string(index=False))

for thr in (1, 31, 93, 155, 310):
    h = nb[nb.b23 >= thr]; g = h[((h.b00 + h.a00) == 0) & (h.last_min >= 24 * 60 + TRANSFER)]
    print(f"민감도 23시승차>={thr}: 쌍 {len(h)}, 강한단절 {len(g)} ({len(g)/max(len(h),1):.1%}), 역 {g.name.nunique()}")
a_cls = home[~home.ends_before_00]
print(f"\n00시대까지 운행하는 귀가방향 노선-정류장: {len(a_cls)}쌍 → 5~15분 조정 여부는 분 단위 막차시각(API) 필요")
