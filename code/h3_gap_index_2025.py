import pandas as pd, numpy as np, glob, re, calendar, os
from scipy.stats import spearmanr
D = "data/"; MON = D + "버스_정류장별시간대별승하차_2025월별/"

def rd(p, **kw):
    for enc in ("utf-8-sig", "cp949"):
        try: return pd.read_csv(p, encoding=enc, **kw)
        except UnicodeDecodeError: pass

home = pd.read_csv("out/policy_home_pairs_2025.csv", dtype={"ars": str, "노선번호": str})
home["ars"] = home["ars"].str.zfill(5)
st = home.groupby("name").apply(lambda g: pd.Series({
    "S1_after00": g.loc[g["mode"].isin(["A1", "A2"]), "노선번호"].nunique(),
    "S2_after01": g.loc[g["mode"] == "A1", "노선번호"].nunique(),
    "n_home_routes": g["노선번호"].nunique(),
    "n_B": g.loc[g["mode"] == "B", "노선번호"].nunique()}), include_groups=False).reset_index()

near = home[["name", "ars"]].drop_duplicates()
allnear = pd.read_csv("out/near_route_stop_class.csv", dtype={"ars": str})[["name", "ars"]].drop_duplicates()
allnear["ars"] = allnear["ars"].str.zfill(5)
near = pd.concat([near, allnear]).drop_duplicates()
nights = []
for p in sorted(glob.glob(MON + "*.csv")):
    b = rd(p, dtype={"버스정류장ARS번호": str})
    b = b[b["교통수단타입명"] == "서울심야버스"].copy()
    b["ars"] = b["버스정류장ARS번호"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    b["n13"] = b[[f"{h}시{k}총승객수" for h in (1, 2, 3) for k in ("승차", "하차")]].sum(axis=1)
    x = near.merge(b[b.n13 > 0][["ars", "노선번호"]], on="ars"); x["ym"] = re.search(r"(2025\d\d)", p).group(1)
    nights.append(x)
N = pd.concat(nights, ignore_index=True).groupby(["name", "노선번호"]).ym.nunique().reset_index()
S3 = N[N.ym >= 6].groupby("name").노선번호.nunique().rename("S3_night")

pol = pd.read_csv("out/policy_station_mix_2025.csv")
sl = pd.read_csv("out/station_summary_v2.csv")[["name", "last_min"]]
df = pol[["name", "alight_24p"]].merge(st, on="name", how="left").merge(S3, on="name", how="left").merge(sl, on="name", how="left")
df[["S1_after00", "S2_after01", "S3_night", "n_B", "n_home_routes"]] = df[["S1_after00", "S2_after01", "S3_night", "n_B", "n_home_routes"]].fillna(0)
df = df.dropna(subset=["alight_24p", "last_min"])
z = lambda s: (s - s.mean()) / s.std(ddof=0)
df["zD"] = z(df.alight_24p)
bus = (z(df.S1_after00) + z(df.S2_after01) + z(df.S3_night)) / 3
df["zS"] = (z(df.S1_after00) + z(df.S2_after01) + z(df.S3_night) + z(df.last_min)) / 4
df["gap"] = df.zD - df.zS
df["gap_ratio"] = df.alight_24p / (df.S1_after00 + df.S3_night + 1)
df["gap_after01"] = df.zD - (z(df.S2_after01) + z(df.S3_night) + z(df.last_min)) / 3
df["gap_bus2"] = df.zD - (2 * bus + z(df.last_min)) / 3
df = df.sort_values("gap", ascending=False).reset_index(drop=True); df["rank"] = df.index + 1
df["막차hhmm"] = df.last_min.map(lambda m: f"{int(m//60)%24:02d}:{int(m%60):02d}")
print(f"역 {len(df)}곳 | 사막(gap>0, 수요가 공급보다 큼) {int((df.gap>0).sum())}곳")
top = set(df.head(10).name)
for c in ("gap_ratio", "gap_after01", "gap_bus2"):
    o = set(df.sort_values(c, ascending=False).head(10).name)
    print(f"민감도 {c}: 상위10 겹침 {len(top & o)}/10 | Spearman {spearmanr(df.gap, df[c]).statistic:.2f}")
ranks = pd.DataFrame({c: df[c].rank(ascending=False) for c in ("gap", "gap_ratio", "gap_after01", "gap_bus2")})
df["top10_count"] = (ranks <= 10).sum(axis=1); df["worst_rank"] = ranks.max(axis=1)
cols = ["rank", "name", "alight_24p", "S1_after00", "S2_after01", "S3_night", "n_B", "막차hhmm", "zD", "zS", "gap", "top10_count", "worst_rank"]
print(df[cols].head(20).round(2).to_string(index=False))
print("\n4개 방법 모두 상위10:", df[df.top10_count == 4].name.tolist())
print("3개 이상 상위10:", df[df.top10_count >= 3].name.tolist())
df.to_csv("out/gap_index_station_2025.csv", index=False, encoding="utf-8-sig")
