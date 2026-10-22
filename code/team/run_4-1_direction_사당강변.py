import os, sys, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from common_direction import load_sequence, load_basic_info, load_stops, build_legs, station_directions, find_turning_point, haversine_m

D = "data/"
STATIONS = ["사당", "강변"]
RADIUS = 300

seq = load_sequence(D + "버스_노선별경유정류소순서_20251209.xlsx")
binfo = load_basic_info(D + "버스_노선기본정보_기점첫막차_20251209.xlsx")
stops_xy = load_stops(D + "버스_정류소위치_20251209.xlsx")
legs = build_legs(seq, binfo)

c18 = pd.read_csv(D + "지하철_역좌표_1-8호선_20250814.csv", encoding="cp949")
norm = lambda n: (lambda s: s[:-1] if len(s) > 2 and s.endswith("역") else s)(re.sub(r"\(.*?\)", "", str(n)).strip())
c18["name"] = c18["역명"].map(norm)
st_xy = c18.groupby("name")[["위도", "경도"]].mean()

home = pd.read_excel("out/8개역_귀가방향노선_2025월별판정.xlsx", dtype={"정류소ARS": str, "노선": str})
home_keys = set(zip(home["역"], home["노선"], home["정류소ARS"].str.zfill(5)))

tables = []
for stn in STATIONS:
    df = station_directions(stn, st_xy.loc[stn, "위도"], st_xy.loc[stn, "경도"], stops_xy, seq, legs, radius=RADIUS)
    df["귀가방향(2025)"] = [(stn, r, a) in home_keys for r, a in zip(df["노선"], df["정류소ARS"])]
    diff = {}
    for rid in df["노선ID"].unique():
        g = legs[rid].stops
        far, _ = find_turning_point(g, None)
        diff[rid] = "" if far == legs[rid].turn_idx else f"최원점 순번 {int(g['seq'].iloc[far])}"
    df["최원점 근사와 차이"] = df["노선ID"].map(diff)
    tables.append(df)
    print(f"[{stn}] 정류소 {df['정류소ARS'].nunique()}곳, 노선 {df['노선'].nunique()}개, (정류소,노선,방향) {len(df)}행 | 귀가방향 {int(df['귀가방향(2025)'].sum())}행"
          f" | 회차점: 종점명 일치 {int((df.drop_duplicates('노선ID')['회차점판정'] == '종점명 일치').sum())}개 노선, 근사 {int((df.drop_duplicates('노선ID')['회차점판정'] == '최원점 근사').sum())}개")
    print("   회차점이 근사 방식과 다른 노선:", sorted(df.loc[df['최원점 근사와 차이'] != '', '노선'].unique().tolist()))
out = pd.concat(tables, ignore_index=True)
os.makedirs("out/team", exist_ok=True)
with pd.ExcelWriter("out/team/4-1_사당강변_노선방향표.xlsx") as w:
    out.to_excel(w, sheet_name="노선방향표", index=False)
    pd.DataFrame({"컬럼": ["방향번호", "방향구간", "방면", "하류정류소수", "종점정류소", "회차점판정", "귀가방향(2025)", "최원점 근사와 차이"],
                  "설명": ["1=기점→종점(회차점까지), 2=종점→기점", "방향번호의 뜻", "그 방향으로 갔을 때 마지막 정류소(회차점 또는 종점)",
                         "역 정류소 다음부터 방향 끝까지의 정류소 수", "True면 이 정류소가 방향의 마지막(하류 없음)",
                         "'종점명 일치'(기본정보 종점명으로 찾음) / '최원점 근사'(기점에서 가장 먼 정류소)",
                         "2025년 23시 승차 일평균 1명 이상인 정류소-노선(팀 정의로 바꿔도 됨)",
                         "지난 근사 방식으로 계산한 회차점 순번(다르면 표시)"]}).to_excel(w, sheet_name="컬럼설명", index=False)
print("저장: out/team/4-1_사당강변_노선방향표.xlsx |", len(out), "행")
