import os, re, glob, calendar, pickle
import numpy as np, pandas as pd

D = "data/"
os.makedirs("out/final", exist_ok=True)

def rd(p, **kw):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(p, encoding=enc, **kw)
        except UnicodeDecodeError:
            pass

def norm(n):
    n = re.sub(r"\(.*?\)", "", str(n)).strip()
    return n[:-1] if len(n) > 2 and n.endswith("역") else n

def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(x))

c18 = rd(D + "지하철_역좌표_1-8호선_20250814.csv").rename(columns={"고유역번호(외부역코드)": "code"})
c18["code"] = [f"{int(v):04d}" for v in c18["code"]]
c18["name"] = c18["역명"].map(norm)
code2name = dict(zip(c18.code, c18.name))
st_xy = c18.groupby("name")[["위도", "경도"]].mean().reset_index()
d = rd(D + "지하철_역별일별시간대별승하차_2025.csv")
d["date"] = pd.to_datetime(d["수송일자"])
hol = pd.to_datetime(["2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-03", "2025-05-05", "2025-05-06",
                      "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-12-25"])
wk = d[(d.date.dt.weekday < 5) & (~d.date.isin(hol)) & (d["승하차구분"] == "하차")].copy()
ndays = wk.date.nunique()
wk["name"] = [code2name.get(f"{int(v):04d}") for v in wk["역번호"]]
daily = wk.groupby(["name", "date"])["24시이후"].sum().reset_index()
dem = daily.groupby("name")["24시이후"].agg(D_mean="mean", D_median="median").reset_index()
tot24 = dem.D_mean.sum()

x = pd.read_excel(D + "버스_정류소위치_20251209.xlsx", dtype={"ARS_ID": str})
x = x[x["정류소타입"] != "한강선착장"].copy()
x["ars"] = x["ARS_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
dist = hav(st_xy["위도"].values[:, None], st_xy["경도"].values[:, None], x["Y좌표"].values[None, :], x["X좌표"].values[None, :])
ii, jj = np.where(dist <= 300)
near = pd.DataFrame({"name": st_xy["name"].values[ii], "ars": x["ars"].values[jj]}).drop_duplicates()

pairs, routes, nights = [], [], []
for p in sorted(glob.glob(D + "버스_정류장별시간대별승하차_2025월별/*.csv")):
    b = rd(p, dtype={"버스정류장ARS번호": str, "교통수단타입코드": str})
    b["ars"] = b["버스정류장ARS번호"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    ym = str(b["사용년월"].iloc[0]); days = calendar.monthrange(int(ym[:4]), int(ym[4:]))[1]
    b["type"] = b["교통수단타입명"].str.replace("서울", "").str.replace("버스", "")
    b["노선번호"] = b["노선번호"].astype(str)
    b["b23"] = b["23시승차총승객수"]; b["b00"] = b["00시승차총승객수"]
    b["h00"] = b["00시승차총승객수"] + b["00시하차총승객수"]; b["h01"] = b["1시승차총승객수"] + b["1시하차총승객수"]
    b["n13"] = b[[f"{h}시{k}총승객수" for h in (1, 2, 3) for k in ("승차", "하차")]].sum(axis=1)
    reg = b[b["type"].isin(["간선", "지선", "마을", "순환"])]
    g = reg.groupby(["ars", "노선번호", "type"])[["b23", "h00", "b00", "h01"]].sum().reset_index()
    g = near.merge(g, on="ars"); g["ym"] = ym; g["days"] = days
    g["cls"] = np.select([g.h01 > 0, g.h00 > 0], ["A1", "A2"], "B")
    pairs.append(g)
    r = reg.groupby(["노선번호", "type"])[["b00", "b23"]].sum().reset_index(); r["days"] = days
    routes.append(r)
    n = b[b["type"] == "심야"].groupby(["ars", "노선번호"]).n13.sum().reset_index(); n["ym"] = ym
    nights.append(near.merge(n[n.n13 > 0], on="ars"))
P = pd.concat(pairs, ignore_index=True); Rt = pd.concat(routes, ignore_index=True); N = pd.concat(nights, ignore_index=True)
mh = P[P.b23 >= P.days]
month_share = pd.crosstab(mh.ym, mh.cls, normalize="index")

key = ["name", "ars", "노선번호", "type"]
ann = P.groupby(key).agg(b23=("b23", "sum"), b00=("b00", "sum"), days=("days", "sum"), months=("ym", "nunique"),
                         mode=("cls", lambda s: s.value_counts().index[0]), stab=("cls", lambda s: s.value_counts().iloc[0] / len(s))).reset_index()
home = ann[(ann.b23 / ann.days >= 1) & (ann.months >= 6)].copy()
type_tab = home.groupby(["type", "mode"]).size().unstack(fill_value=0)

rank = {"A1": 0, "A2": 1, "B": 2}
sr = home.groupby(["name", "노선번호", "type"]).agg(b23=("b23", "sum"), cls=("mode", lambda s: min(s, key=rank.get))).reset_index()
sr = sr.merge(home.groupby("name").b23.sum().rename("st_b23"), on="name").merge(dem, on="name", how="left")
sr["link"] = sr.D_mean * sr.b23 / sr.st_b23
Ra = Rt.groupby(["노선번호", "type"]).agg(b00=("b00", "sum"), b23=("b23", "sum"), days=("days", "sum")).reset_index()
Ra["b00_day"] = Ra.b00 / Ra.days
late = Ra[Ra.b00_day >= 1]
bench = late.b00_day.median()
A2 = sr[sr.cls == "A2"]; B = sr[sr.cls == "B"]
Br = B.groupby(["노선번호", "type"]).agg(link=("link", "sum"), n_st=("name", "nunique"), stations=("name", lambda s: "·".join(sorted(s)[:4]))).reset_index()
Br["tier"] = np.where(Br.link >= bench, "1회 추가", "대체수단")
Br = Br.sort_values("link", ascending=False).reset_index(drop=True)
A2r = A2.groupby(["노선번호", "type"]).link.sum().reset_index()
Nn = N.groupby(["name", "노선번호"]).ym.nunique().reset_index(); Nn = Nn[Nn.ym >= 6]
S3 = Nn.groupby("name").노선번호.nunique().rename("S3")
Bst = B.groupby("name").link.sum().reset_index().merge(S3, on="name", how="left")
alt = pd.DataFrame([
    {"대안": "① 00시대 종료 노선 5~15분 조정", "노선": A2.노선번호.nunique(), "역-노선": len(A2), "역": A2.name.nunique(), "link": A2.link.sum(), **A2.drop_duplicates("노선번호").type.value_counts().to_dict()},
    {"대안": "② 자정 전 종료 노선 막차 1회 추가", "노선": int((Br.tier == "1회 추가").sum()), "역-노선": int(B.노선번호.isin(Br[Br.tier == "1회 추가"].노선번호).sum()), "역": B[B.노선번호.isin(Br[Br.tier == "1회 추가"].노선번호)].name.nunique(),
     "link": Br[Br.tier == "1회 추가"].link.sum(), **Br[Br.tier == "1회 추가"].type.value_counts().to_dict()},
    {"대안": "③ 저수요 노선 대체수단(올빼미·DRT) 연계", "노선": int((Br.tier == "대체수단").sum()), "역-노선": int(B.노선번호.isin(Br[Br.tier == "대체수단"].노선번호).sum()), "역": B[B.노선번호.isin(Br[Br.tier == "대체수단"].노선번호)].name.nunique(),
     "link": Br[Br.tier == "대체수단"].link.sum(), **Br[Br.tier == "대체수단"].type.value_counts().to_dict()},
]).fillna(0)

stt = home.groupby("name").apply(lambda g: pd.Series({
    "S1": g.loc[g["mode"].isin(["A1", "A2"]), "노선번호"].nunique(), "S2": g.loc[g["mode"] == "A1", "노선번호"].nunique(),
    "nB": g.loc[g["mode"] == "B", "노선번호"].nunique(), "n_home": g["노선번호"].nunique()}), include_groups=False).reset_index()
ST = dem.merge(stt, on="name", how="inner").merge(S3, on="name", how="left").fillna({"S3": 0})
ST["ratio"] = ST.D_median / (ST.S1 + ST.S3 + 1)
z = lambda s: (s - s.mean()) / s.std(ddof=0)
ST["gap_z"] = z(ST.D_mean) - (z(ST.S1) + z(ST.S2) + z(ST.S3)) / 3
qD, qR = ST.D_median.quantile(.75), ST.ratio.quantile(.75)
ST["quad"] = (ST.D_median >= qD) & (ST.ratio >= qR)
ST = ST.sort_values("ratio", ascending=False).reset_index(drop=True)
Q = ST[ST.quad].reset_index(drop=True); Q["qrank"] = Q.index + 1
ST["event"] = ST.D_mean / ST.D_median
top_gap = ST.sort_values("gap_z", ascending=False).name.head(10).tolist()
top_ratio = ST.sort_values("ratio", ascending=False).name.head(10).tolist()

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "team"))
from common_direction import load_sequence
seqf = load_sequence(D + "버스_노선별경유정류소순서_20251209.xlsx")
bi = pd.read_excel(D + "버스_노선기본정보_기점첫막차_20251209.xlsx", sheet_name="평일공동배차", dtype={"노선번호": str}).drop_duplicates("노선번호").set_index("노선번호")

def tm_(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    h, m_ = [int(x) for x in str(v).split(":")[:2]]
    return (h + 24 if h < 4 else h) * 60 + m_

pos = {}
for rid, g in seqf.groupby("route_id"):
    nm = g["route_name"].iloc[0]; n = len(g)
    for i, (a_, sq) in enumerate(zip(g["ars"].values, g["seq"].values)):
        pos.setdefault((nm, a_), []).append((int(sq), n))
chk = []
xt_excl = {}
for _, r in home.iterrows():
    rt = r["노선번호"]
    if rt not in bi.index:
        xt_excl[(r["mode"], "노선기본정보(평일공동배차)에 없음")] = xt_excl.get((r["mode"], "노선기본정보(평일공동배차)에 없음"), 0) + 1
        continue
    if (rt, r["ars"]) not in pos:
        xt_excl[(r["mode"], "경유정류소 순번에 없음")] = xt_excl.get((r["mode"], "경유정류소 순번에 없음"), 0) + 1
        continue
    o = tm_(bi.loc[rt, "막차시간"]); run = bi.loc[rt, "운행시간"]
    if np.isnan(o):
        xt_excl[(r["mode"], "기점 막차시간 결측")] = xt_excl.get((r["mode"], "기점 막차시간 결측"), 0) + 1
        continue
    ts = [o + (sq / n * run * 0.85 if pd.notna(run) else sq * 1.45) for sq, n in pos[(rt, r["ars"])]]
    t_last = max(ts)
    chk.append({"mode": r["mode"], "model_cls": "A1" if t_last >= 25 * 60 else ("A2" if t_last >= 24 * 60 else "B")})
chk = pd.DataFrame(chk)
XT = pd.crosstab(chk["mode"], chk["model_cls"]).reindex(index=["A1", "A2", "B"], columns=["A1", "A2", "B"]).fillna(0).astype(int)
agree = {k: XT.loc[k, k] / XT.loc[k].sum() for k in ["A1", "A2", "B"]}
print("시간표 교차검증(행=이용기록 등급, 열=시간표 추정 등급):"); print(XT); print("등급별 일치율:", {k: round(v, 3) for k, v in agree.items()}, "| 대상", len(chk))
print("B(이용기록 없음) 중 시간표상 자정 이후 통과:", int(XT.loc["B", ["A1", "A2"]].sum()), f"({XT.loc['B', ['A1', 'A2']].sum() / XT.loc['B'].sum():.1%})")
n_home_mode = home["mode"].value_counts().to_dict()
print("교차검증 제외 사유:", xt_excl, "| 등급별 전체", n_home_mode)

pickle.dump({"XT": XT, "agree": agree, "n_chk": len(chk), "xt_excl": xt_excl, "n_home_mode": n_home_mode, "ST": ST, "Q": Q, "qD": qD, "qR": qR, "alt": alt, "Br": Br, "A2r": A2r, "sr": sr, "bench": bench, "n_late": len(late), "type_tab": type_tab,
             "month_share": month_share, "home_n": len(home), "home_st": home.name.nunique(), "stab": home.stab.mean(), "tot24": tot24, "ndays": ndays,
             "Bst": Bst, "top_gap": top_gap, "top_ratio": top_ratio, "n_dem": len(dem)}, open("out/final/_citywide.pkl", "wb"))
ST.to_csv("out/final/역별_수요공급_2025.csv", index=False, encoding="utf-8-sig")

print(f"수요 자료 역 {len(dem)} | 귀가방향 노선 있는 분석 대상 역 {len(ST)} | 24시 이후 하차 합 {tot24:,.0f}명/평일(일수 {ndays})")
print(f"귀가방향 노선-정류장 {len(home)}쌍 / 역 {home.name.nunique()} | 최빈 등급 일치율 {home.stab.mean():.3f}")
print(type_tab.assign(합계=type_tab.sum(axis=1), B비중=(type_tab["B"] / type_tab.sum(axis=1)).round(3)))
print("등급 비중:", home["mode"].value_counts(normalize=True).round(3).to_dict())
print(f"기준선: 00시대 운행 일반노선 {len(late)}개 중앙값 {bench:.1f}명")
print(alt.round(0).to_string(index=False))
print(f"B 보유 역 {len(Bst)} 중 올빼미 보유 {int(Bst.S3.notna().sum())} ({Bst.S3.notna().mean():.1%}) | 미보유 역의 연계수요 {Bst.loc[Bst.S3.isna(), 'link'].sum():,.0f}")
print(f"선정 기준: D_median ≥ {qD:.0f}, ratio ≥ {qR:.1f} → {len(Q)}곳")
print(Q[["qrank", "name", "D_median", "S1", "S2", "S3", "nB", "ratio"]].head(12).round(1).to_string(index=False))
print("차이법 top10:", top_gap); print("비율법 top10:", top_ratio, "| 겹침", [s for s in top_gap if s in top_ratio])
print("행사성(평균/중앙값>1.3):", ST[ST.event > 1.3][["name", "event"]].round(2).values.tolist())
print("월별 등급 비중(1,6,12월):", month_share.loc[["202501", "202506", "202512"]].round(3).to_dict("index"))
print("B 1회 추가 상위 10:", Br.head(10)[["노선번호", "type", "link", "stations"]].round(0).values.tolist())
