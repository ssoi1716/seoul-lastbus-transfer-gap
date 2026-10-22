import pandas as pd, numpy as np, re
D = "data/"
c18 = pd.read_csv(D + "지하철_역좌표_1-8호선_20250814.csv", encoding="cp949").rename(columns={"고유역번호(외부역코드)": "code"})
norm = lambda n: (lambda s: s[:-1] if len(s) > 2 and s.endswith("역") else s)(re.sub(r"\(.*?\)", "", str(n)).strip())
code2name = {f"{int(c):04d}": norm(n) for c, n in zip(c18.code, c18.역명)}
d = pd.read_csv(D + "지하철_역별일별시간대별승하차_2025.csv", encoding="cp949")
d["date"] = pd.to_datetime(d["수송일자"])
hol = pd.to_datetime(["2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-03", "2025-05-05", "2025-05-06",
                      "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-12-25"])
w = d[(d.date.dt.weekday < 5) & (~d.date.isin(hol)) & (d["승하차구분"] == "하차")].copy()
w["name"] = [code2name.get(f"{int(v):04d}") for v in w["역번호"]]
daily = w.groupby(["name", "date"])["24시이후"].sum().reset_index()
med = daily.groupby("name")["24시이후"].agg(D_mean="mean", D_median="median").reset_index()
df = pd.read_csv("out/gap_index_station_2025.csv").merge(med, on="name", how="left")
df["event_ratio"] = df.D_mean / df.D_median
df["ratio_med"] = df.D_median / (df.S1_after00 + df.S3_night + 1)
qD, qR = df.D_median.quantile(.75), df.ratio_med.quantile(.75)
q = df[(df.D_median >= qD) & (df.ratio_med >= qR)].sort_values("ratio_med", ascending=False)
q.to_csv("out/gap_quadrant_candidates_2025.csv", index=False, encoding="utf-8-sig")
print(f"기준 D≥{qD:.0f}, 비율≥{qR:.1f} → {len(q)}곳"); print(q[["name", "D_median", "S1_after00", "S3_night", "ratio_med"]].head(10).round(1).to_string(index=False))
