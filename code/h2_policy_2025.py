import pandas as pd, numpy as np, re, os, glob, calendar
os.makedirs("out", exist_ok=True)
D = "data/"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

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

c18 = rd(D + "지하철_역좌표_1-8호선_20250814.csv").rename(columns={"고유역번호(외부역코드)": "code"})
c9 = rd(D + "지하철_역좌표_9호선2-3단계_20260131.csv").rename(columns={"역번호": "code"})
coords = pd.concat([c18[["호선", "code", "역명", "위도", "경도"]], c9[["호선", "code", "역명", "위도", "경도"]]], ignore_index=True)
coords["code"] = [code4(v) for v in coords["code"]]
coords["name"] = [norm(v) for v in coords["역명"]]
coords["호선"] = coords["호선"].astype(str)
code2name = dict(zip(coords.code, coords.name))

t = rd(D + "지하철_열차운행시각표_20260902.csv", dtype=str)
t = t[t["주중주말"] == "DAY"].copy()
t["arr"] = t["열차도착시간"].map(tmin)
t = t.dropna(subset=["arr"])
t["name_tt"] = t["역사명"].map(norm)
t18 = t[t["호선"] != "9"].copy(); t18["name"] = t18["역사코드"].map(code2name)
names9 = set(coords.loc[coords["호선"] == "9", "name"])
t9 = t[(t["호선"] == "9") & (t["name_tt"].isin(names9))].copy(); t9["name"] = t9["name_tt"]
tt = pd.concat([t18.dropna(subset=["name"]), t9])
st_last = tt.groupby(["name", "호선", "방향"])["arr"].max().groupby("name").agg(last_min="min", last_max="max").reset_index()

d = rd(D + "지하철_역별일별시간대별승하차_2025.csv")
d["date"] = pd.to_datetime(d["수송일자"])
hol = pd.to_datetime(["2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-03",
                      "2025-05-05", "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03",
                      "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-12-25"])
wk = d[(d.date.dt.weekday < 5) & (~d.date.isin(hol)) & (d["승하차구분"] == "하차")].copy()
ndays = wk.date.nunique()
wk["name"] = [code2name.get(code4(v)) for v in wk["역번호"]]
dem = (wk.groupby("name")[["24시이후"]].sum() / ndays).rename(columns={"24시이후": "alight_24p"}).reset_index()

x = pd.read_excel(D + "버스_정류소위치_20260902.xlsx", dtype={"ARS_ID": str})
x = x[x["정류소타입"] != "한강선착장"].copy()
x["ars"] = x["ARS_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
st_xy = coords.groupby("name")[["위도", "경도"]].mean().reset_index()
lat1 = np.radians(st_xy["위도"].values)[:, None]; lon1 = np.radians(st_xy["경도"].values)[:, None]
lat2 = np.radians(x["Y좌표"].values)[None, :]; lon2 = np.radians(x["X좌표"].values)[None, :]
a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
dist = 2 * 6371000 * np.arcsin(np.sqrt(a))
ii, jj = np.where(dist <= 300)
near = pd.DataFrame({"name": st_xy["name"].values[ii], "ars": x["ars"].values[jj]}).drop_duplicates()

pairs, routes, nights = [], [], []
for p in sorted(glob.glob(D + "버스_정류장별시간대별승하차_2025월별/버스_정류장별시간대별승하차_2025*.csv")):
    b = rd(p, dtype={"버스정류장ARS번호": str, "교통수단타입코드": str})
    b["ars"] = b["버스정류장ARS번호"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    ym = str(b["사용년월"].iloc[0]); days = calendar.monthrange(int(ym[:4]), int(ym[4:]))[1]
    b["type"] = b["교통수단타입명"].str.replace("서울", "").str.replace("버스", "")
    b["노선번호"] = b["노선번호"].astype(str)
    b["b23"] = b["23시승차총승객수"]; b["h00"] = b["00시승차총승객수"] + b["00시하차총승객수"]
    b["b00"] = b["00시승차총승객수"]; b["h01"] = b["1시승차총승객수"] + b["1시하차총승객수"]
    b["n13"] = b[["1시승차총승객수", "2시승차총승객수", "3시승차총승객수", "1시하차총승객수", "2시하차총승객수", "3시하차총승객수"]].sum(axis=1)
    reg = b[b["type"].isin(["간선", "지선", "마을", "순환"])]
    g = reg.groupby(["ars", "노선번호", "type"])[["b23", "h00", "b00", "h01"]].sum().reset_index()
    g = near.merge(g, on="ars"); g["ym"] = ym; g["days"] = days
    g["cls"] = np.select([g.h01 > 0, g.h00 > 0], ["A1", "A2"], "B")
    pairs.append(g)
    r = reg.groupby(["노선번호", "type"])[["b00", "b23"]].sum().reset_index(); r["days"] = days; r["ym"] = ym
    routes.append(r)
    n = b[b["type"] == "심야"].groupby(["ars", "노선번호"]).n13.sum().reset_index(); n["ym"] = ym
    nights.append(near.merge(n[n.n13 > 0], on="ars"))
P = pd.concat(pairs, ignore_index=True); R = pd.concat(routes, ignore_index=True); N = pd.concat(nights, ignore_index=True)
print("월 파일:", P.ym.nunique())

mh = P[P.b23 >= P.days]
print("\n월별 등급 비중(귀가방향 쌍):")
print(pd.crosstab(mh.ym, mh.cls, normalize="index").round(3))

key = ["name", "ars", "노선번호", "type"]
ann = P.groupby(key).agg(b23=("b23", "sum"), b00=("b00", "sum"), days=("days", "sum"), months=("ym", "nunique"),
                         mode=("cls", lambda s: s.value_counts().index[0]),
                         stab=("cls", lambda s: s.value_counts().iloc[0] / len(s))).reset_index()
home = ann[(ann.b23 / ann.days >= 1) & (ann.months >= 6)].copy()
print(f"\n[연간 최빈 등급] 귀가방향 노선-정류장 {len(home)}쌍 / 역 {home.name.nunique()}")
tab = home.groupby(["type", "mode"]).size().unstack(fill_value=0); tab["합계"] = tab.sum(axis=1)
print(tab); print(home["mode"].value_counts(normalize=True).round(3).to_dict())
print("최빈 등급 월 일치율(평균):", home.groupby("mode").stab.mean().round(3).to_dict(), "| 전체", round(home.stab.mean(), 3))
print("마을버스 B 비중:", round((home[home.type == "마을"]["mode"] == "B").mean(), 3),
      "| 지선:", round((home[home.type == "지선"]["mode"] == "B").mean(), 3), "| 간선:", round((home[home.type == "간선"]["mode"] == "B").mean(), 3))

rank = {"A1": 0, "A2": 1, "B": 2}
sr = home.groupby(["name", "노선번호", "type"]).agg(b23=("b23", "sum"), b00=("b00", "sum"), days=("days", "max"),
                                                   cls=("mode", lambda s: min(s, key=rank.get))).reset_index()
sr = sr.merge(home.groupby("name").b23.sum().rename("st_b23"), on="name").merge(dem, on="name", how="left").merge(st_last, on="name", how="left")
sr["share"] = sr.b23 / sr.st_b23
sr["link"] = sr.alight_24p * sr.share
print("\n역-노선 등급:", sr.cls.value_counts().to_dict(), "| 고유노선:", sr.groupby("cls").노선번호.nunique().to_dict())

Ra = R.groupby(["노선번호", "type"]).agg(b00=("b00", "sum"), b23=("b23", "sum"), days=("days", "sum")).reset_index()
Ra["b00_day"] = Ra.b00 / Ra.days; Ra["b23_day"] = Ra.b23 / Ra.days
late = Ra[Ra.b00_day >= 1]
thr = late.b00_day.median()
print(f"\n[기준선] 00시대 운행 일반노선 {len(late)}개의 노선 전체 00시대 1일 승차: 중앙값 {thr:.1f}명 (사분위 {late.b00_day.quantile(.25):.1f}~{late.b00_day.quantile(.75):.1f})")
print("  유형별 중앙값:", late.groupby("type").b00_day.median().round(1).to_dict())

B = sr[sr.cls == "B"]
Br = B.groupby(["노선번호", "type"]).agg(link=("link", "sum"), n_st=("name", "nunique"),
                                          stations=("name", lambda s: "·".join(sorted(s)[:4]))).reset_index()
Br = Br.merge(Ra[["노선번호", "type", "b23_day"]], on=["노선번호", "type"], how="left")
Br["tier"] = np.where(Br.link >= thr, "1회 추가 우선", "대체수단 연계")
Br = Br.sort_values("link", ascending=False).reset_index(drop=True)
tot = Br.link.sum(); Br["cum"] = Br.link.cumsum() / tot
print(f"\n[B 자정 전 종료] 노선 {len(Br)}개 | 환승 연계수요 합계 {tot:,.0f}명/평일")
tb = Br.groupby(["tier", "type"]).size().unstack(fill_value=0); tb["합계"] = tb.sum(axis=1); print(tb)
print("  tier별 연계수요:", Br.groupby("tier").link.sum().round(0).to_dict())
for k in (10, 20, 30, 50):
    print(f"  상위 {k}개 노선 1회 추가 → 연계수요 {Br.cum.iloc[k-1]:.1%} ({Br.link.head(k).sum():,.0f}명/평일)")
print(f"  연계수요 50% 도달 노선 수: {int((Br.cum < .5).sum()) + 1}")
print("  1회 추가 우선 노선 유형 비중:", Br[Br.tier == "1회 추가 우선"].type.value_counts(normalize=True).round(3).to_dict())
print("\n1회 추가 우선 상위 15개 노선:")
print(Br.head(15)[["노선번호", "type", "link", "n_st", "stations", "b23_day"]].round(1).to_string(index=False))

A2 = sr[sr.cls == "A2"]
print(f"\n[A2 00시대 종료] 역-노선 {len(A2)}개 · 고유 노선 {A2.노선번호.nunique()} · 역 {A2.name.nunique()} | 유형 {A2.type.value_counts().to_dict()}")
print(f"  소속 역 막차 최종도착 00:45 이후: {(A2.last_max >= 24*60+45).mean():.1%}")
print(f"  A2 역-노선 환승 연계수요 합계 {A2.link.sum():,.0f}명/평일")

Nn = N.groupby(["name", "노선번호"]).ym.nunique().reset_index()
Nn = Nn[Nn.ym >= 6]
ns = Nn.groupby("name").노선번호.apply(lambda s: ", ".join(sorted(set(s)))).rename("night")
Bst = B.groupby("name").agg(nB=("노선번호", "nunique"), linkB=("link", "sum")).reset_index().merge(ns, on="name", how="left")
Bst["has_night"] = Bst.night.notna()
low_routes = set(Br[Br.tier == "대체수단 연계"].노선번호)
Bst["has_low"] = Bst.name.isin(B[B.노선번호.isin(low_routes)].name)
print(f"\n[대체수단] B 보유 역 {len(Bst)}곳 중 300m 내 올빼미버스(01~03시 활동, 6개월+) 보유 {Bst.has_night.sum()}곳 ({Bst.has_night.mean():.1%})")
print(f"  대체수단 연계 노선이 있는 역 {Bst.has_low.sum()}곳 중 올빼미 미보유 {(Bst.has_low & ~Bst.has_night).sum()}곳")
print("  올빼미 미보유 역 B 연계수요 상위 10:")
print(Bst[~Bst.has_night].sort_values("linkB", ascending=False).head(10)[["name", "nB", "linkB"]].round(1).to_string(index=False))
print(f"  올빼미 미보유 역의 B 연계수요 합계 {Bst.loc[~Bst.has_night, 'linkB'].sum():,.0f}명 ({Bst.loc[~Bst.has_night, 'linkB'].sum()/Bst.linkB.sum():.1%})")

Br.to_csv("out/policy_B_routes_2025.csv", index=False, encoding="utf-8-sig")
A2.to_csv("out/policy_A2_station_routes_2025.csv", index=False, encoding="utf-8-sig")
Bst.to_csv("out/policy_B_station_night_2025.csv", index=False, encoding="utf-8-sig")
home.to_csv("out/policy_home_pairs_2025.csv", index=False, encoding="utf-8-sig")
