import os, math, html, base64, pickle
import numpy as np, pandas as pd

FONT = os.path.expanduser("~/Downloads/Pretendard-1.3.9/web/static/woff2/")
OUT = "최종산출물_사당강변_막차연계_분석보고서.html"
d = pickle.load(open("out/final/_final.pkl", "rb"))
R, H, M, TP, S = d["R"], d["H"], d["M"], d["TP"], d["S"]
st_last, last_dir, late, after0, st_xy, full_order = d["st_last"], d["last_dir"], d["late"], d["after0"], d["st_xy"], d["full_order"]
nmon = d["nmon"]
CW = pickle.load(open("out/final/_citywide.pkl", "rb"))
CST, CQ, ALT, CBr, csr = CW["ST"], CW["Q"], CW["alt"], CW["Br"], CW["sr"]
n_st, n_q, qD, qR = len(CST), len(CQ), CW["qD"], CW["qR"]
TT = CW["type_tab"].reindex(["간선", "지선", "마을", "순환"]).fillna(0).astype(int)
TRANSFER = 6.0
ST = ["사당", "강변"]
DMED = {"사당": 777.0, "강변": 202.5}

U, MA, EX, EXA, WAIT = d["U"], d["MA"], d["EX"], d["EXA"], d["WAIT"]
XT, AGREE = CW["XT"], CW["agree"]
n_u = len(U); n_conn_u = int((U["gap"] <= 0).sum()); gap_med_u = U["gap"].median()
bench = CW["bench"]

def alt_counts(th):
    hi = CBr[CBr["link"] >= th]; lo = CBr[CBr["link"] < th]
    return len(hi), hi["link"].sum(), len(lo), lo["link"].sum()

BS = [(th, *alt_counts(th)) for th in (bench / 2, bench, bench * 2, 100.0)]
sel_b = {s_: [i[0][0] for i in M[s_]["res"]["selected"]] for s_ in ST}
sel_a = {s_: [i[0][0] for i in MA[s_]["res"]["selected"]] for s_ in ST}
common_sel = {s_: [r_ for r_ in sel_b[s_] if r_ in sel_a[s_]] for s_ in ST}
dep_b = {s_: [r_ for r_ in sel_b[s_] if r_ not in sel_a[s_]] for s_ in ST}
dep_a = {s_: [r_ for r_ in sel_a[s_] if r_ not in sel_b[s_]] for s_ in ST}
n_common = sum(len(common_sel[s_]) for s_ in ST)
FIX = d["FIX"]
XEX = CW.get("xt_excl", {}); NHM = CW.get("n_home_mode", {})
xB = {k_[1]: v_ for k_, v_ in XEX.items() if k_[0] == "B"}
nB_home = int(NHM.get("B", 0)); nB_chk = int(XT.loc["B"].sum()); nB_ex = nB_home - nB_chk
xB_txt = " + ".join(f"{k_} {v_}" for k_, v_ in sorted(xB.items(), key=lambda x: -x[1]))
link_sr = {(r["name"], r["노선번호"]): r["link"] for _, r in csr.iterrows()}
prio = lambda s_, r_: "1순위(공통)" if r_ in common_sel[s_] else "2순위(승차 가중)"
prio_s = lambda s_, r_: "1순위" if r_ in common_sel[s_] else "2순위"

def near_note(ex, target):
    for k_ in sorted(ex):
        if ex[k_][0] >= target:
            if k_ - 1 in ex and round(ex[k_ - 1][0] * 100, 1) >= round(target * 100, 1):
                return f"{k_ - 1}개는 {ex[k_ - 1][0] * 100:.2f}%로 {target * 100:.0f}%에 {(target - ex[k_ - 1][0]) * 100:.2f}%p 미달"
            return ""
    return ""

def need_k(ex, target):
    for k_ in sorted(ex):
        if ex[k_][0] >= target:
            return str(k_)
    return f"{max(ex)}개 초과"

def run_est(r):
    return r["운행시간"] * r["하류정류소수"] / r["총정류소수"] * 0.85 if pd.notna(r["운행시간"]) else r["하류정류소수"] * 1.45

def scn_dir(r):
    m_ = H[(H["역"] == r["역"]) & (H["노선"] == r["노선"]) & (H["방향번호"] == r["방향번호"])]
    return DMED[r["역"]] * m_["share_dir"].sum()

def hhmm(x):
    return "" if x is None or pd.isna(x) else f"{int(x // 60) % 24:02d}:{int(x % 60):02d}"

def f0(x): return f"{x:,.0f}"
def f1(x): return f"{x:,.1f}"
def pct(x, n=1): return f"{x * 100:.{n}f}%"
def esc(s): return html.escape(str(s))

n_R, n_H = len(R), len(H)
v = R.dropna(subset=["api", "model"])
err_mean, err_med, err_mae = v["err"].mean(), v["err"].median(), v["err"].abs().mean()
in10, in20 = (v["err"].abs() <= 10).mean(), (v["err"].abs() <= 20).mean()
worst = v.loc[v["err"].abs().idxmax()]
repro_ok = int((R["repro_diff"] <= 1).sum()); repro_n = int(R["team_model"].notna().sum())
Hrep = H.dropna(subset=["team_model"]); repro_H = int((Hrep["repro_diff"] <= 1).sum())
R["act00_월"] = (R["b00"] + R["a00"]) / nmon
outl = R[(R["model"] < 1440) & (R["act00_월"] > 0)]
outl_big = outl[outl["act00_월"] >= 30]
outl_api_after = int((outl_big["api"] >= 1440).sum())
n_conn = int((H["gap"] <= 0).sum()); n_sure = int((H["판정"] == "연결").sum()); n_edge = int((H["판정"] == "경계(모델 오차 범위)").sum())
gap_med = H["gap"].median(); gap_med_s = H.groupby("역")["gap"].median()
n_le15 = int(((H["gap"] > 0) & (H["gap"] <= 15)).sum()); n_gt30 = int((H["gap"] > 30).sum())
CH = H[H["선택순서"].notna()].sort_values(["역", "선택순서"])
CH = pd.concat([CH[CH["역"] == s] for s in ST])
n_ch = len(CH); type_cnt = CH["유형"].value_counts().to_dict()
cov = {s: (M[s]["res"]["base_cov"], M[s]["res"]["final_cov"], len(M[s]["res"]["selected"])) for s in ST}
gain_ppl = {s: DMED[s] * (cov[s][1] - cov[s][0]) for s in ST}
r5530 = H[(H["노선"] == "5530") & (H["정류소ARS"] == "20253")].iloc[0]
r540 = H[(H["노선"] == "540") & (H["방면"].str.contains("군포"))].iloc[0]
rank5530 = [i for i, (k, g, c) in enumerate(full_order["사당"]) if k[0] == "5530"][0] + 1
gain5530 = [g for k, g, c in full_order["사당"] if k[0] == "5530"][0]
tp5530 = TP[TP["노선"] == "5530"].iloc[0]

def srow(name):
    return S[S["구분"] == name].iloc[0]

_tier = dict(zip(CBr["노선번호"], CBr["tier"]))
_alt = {(r["name"], r["노선번호"]): {"A1": "01시 이후 이용 기록", "A2": "대안① 후보", "B": "대안② 대상" if _tier.get(r["노선번호"]) == "1회 추가" else "대안③ 대상"}[r["cls"]]
        for _, r in csr[csr["name"].isin(ST)].iterrows()}
H["전역분류"] = [_alt.get((a, b), "") for a, b in zip(H["역"], H["노선"])]
CH["전역분류"] = [_alt.get((a, b), "") for a, b in zip(CH["역"], CH["노선"])]
BR = pd.crosstab(H["전역분류"], pd.cut(H["gap"], [-1e9, 0, 15, 1e9], labels=["연결", "15분 이내", "15분 초과"]))
BR = BR.reindex(index=["01시 이후 이용 기록", "대안① 후보", "대안② 대상", "대안③ 대상"], columns=["연결", "15분 이내", "15분 초과"]).fillna(0).astype(int)
a1_n = int(BR.loc["대안① 후보"].sum()); a1_ok = int(BR.loc["대안① 후보", ["연결", "15분 이내"]].sum())

aw, ah = 900, 180
sa = [f'<svg viewBox="0 0 {aw} {ah}" class="chart" role="img" aria-label="정책 대안 3종의 대상 노선 수와 해당 노선 몫 하차 인원">']
names = ["대안① 5~15분 조정", "대안② 막차 1회 추가", "대안③ 대체수단 연계"]
for k, (col, title) in enumerate((("노선", "대상 노선 수(개)"), ("link", "해당 노선 몫 24시 이후 하차(명/평일)"))):
    ox = 10 + k * 450; x0 = ox + 132; wmax = 450 - 132 - 70
    sa.append(f'<text x="{x0}" y="16" class="rowlbl">{title}</text>')
    vmax = ALT[col].max()
    for i, (_, r) in enumerate(ALT.iterrows()):
        y = 32 + i * 46; w_ = max(r[col] / vmax * wmax, 2)
        sa.append(f'<text x="{x0 - 8}" y="{y + 19}" class="tick" text-anchor="end">{names[i]}</text>'
                  f'<rect x="{x0}" y="{y}" width="{w_:.1f}" height="28" rx="3" fill="#111"><title>{names[i]}: {r[col]:,.0f}</title></rect>'
                  f'<text x="{x0 + w_ + 6:.1f}" y="{y + 19}" class="ann strong">{r[col]:,.0f}</text>')
sa.append("</svg>")
FIGA = "".join(sa)

sw, sh, sml, smb = 900, 440, 64, 54
xmn, xmx, ymn, ymx = 10, 2000, 1, 200
SX = lambda v_: sml + (math.log10(min(max(v_, xmn), xmx)) - 1) / (math.log10(xmx) - 1) * (sw - sml - 24)
SY = lambda v_: 16 + (sh - 16 - smb) * (1 - math.log10(min(max(v_, ymn), ymx)) / math.log10(ymx))
ss = [f'<svg viewBox="0 0 {sw} {sh}" class="chart" role="img" aria-label="역별 심야 수요와 이용자당 공급 부족">']
ss.append(f'<rect x="{SX(qD):.1f}" y="16" width="{sw - 24 - SX(qD):.1f}" height="{SY(qR) - 16:.1f}" fill="#efefef"/>')
for t_ in (10, 20, 50, 100, 200, 500, 1000, 2000):
    ss.append(f'<line x1="{SX(t_):.1f}" y1="16" x2="{SX(t_):.1f}" y2="{sh - smb}" stroke="#e6e6e6"/><text x="{SX(t_):.1f}" y="{sh - smb + 18}" class="tick" text-anchor="middle">{t_:,}</text>')
for t_ in (1, 2, 5, 10, 20, 50, 100, 200):
    ss.append(f'<line x1="{sml}" y1="{SY(t_):.1f}" x2="{sw - 24}" y2="{SY(t_):.1f}" stroke="#e6e6e6"/><text x="{sml - 8}" y="{SY(t_) + 4:.1f}" class="tick" text-anchor="end">{t_}</text>')
ss.append(f'<line x1="{SX(qD):.1f}" y1="16" x2="{SX(qD):.1f}" y2="{sh - smb}" stroke="#555" stroke-width="1.5" stroke-dasharray="4 3"/>'
          f'<line x1="{sml}" y1="{SY(qR):.1f}" x2="{sw - 24}" y2="{SY(qR):.1f}" stroke="#555" stroke-width="1.5" stroke-dasharray="4 3"/>'
          f'<text x="{SX(qD) + 6:.1f}" y="{sh - smb - 8}" class="ann">수요 상위 25% ({qD:.0f}명)</text><text x="{sml + 6}" y="{SY(qR) - 6:.1f}" class="ann">이용자당 공급 부족 상위 25% ({qR:.1f})</text>'
          f'<text x="{sw - 32}" y="32" class="ann strong" text-anchor="end">우선 검토 영역 {n_q}곳</text>')
top8 = CQ["name"].head(8).tolist()
off = {"사당": (11, -9, "start"), "강변": (-11, -9, "end"), "창동": (10, -9, "start"), "대림": (10, 16, "start"), "영등포구청": (-11, 6, "end"), "신대방": (10, 5, "start"), "성수": (-11, 22, "end"), "건대입구": (10, 15, "start")}
for _, r in CST.sort_values("quad").iterrows():
    nm = r["name"]; x_, y_ = SX(r["D_median"]), SY(r["ratio"])
    tip = esc(f'{nm}|24시 이후 하차(중앙값) {r["D_median"]:.0f}명|00시 이후 노선 {int(r["S1"])}개 · 올빼미 {int(r["S3"])}개|이용자당 공급 부족 {r["ratio"]:.1f}')
    if nm in ST:
        ss.append(f'<circle class="d-ch" cx="{x_:.1f}" cy="{y_:.1f}" r="8" data-tip="{tip}"/>')
    elif nm in top8:
        ss.append(f'<circle cx="{x_:.1f}" cy="{y_:.1f}" r="5.5" fill="#555" stroke="#fff" stroke-width="1.5" data-tip="{tip}"/>')
    elif r["quad"]:
        ss.append(f'<circle cx="{x_:.1f}" cy="{y_:.1f}" r="4.5" fill="#a8a8a8" stroke="#fff" stroke-width="1.2" data-tip="{tip}"/>')
    else:
        ss.append(f'<circle class="d" cx="{x_:.1f}" cy="{y_:.1f}" r="3.8" data-tip="{tip}"/>')
for nm in top8:
    r = CST[CST["name"] == nm].iloc[0]; dx, dy, anc = off.get(nm, (10, 4, "start"))
    cls_ = "lbl halo big" if nm in ST else "ann halo"
    ss.append(f'<text x="{SX(r["D_median"]) + dx:.1f}" y="{SY(r["ratio"]) + dy:.1f}" class="{cls_}" text-anchor="{anc}">{nm}</text>')
ss.append(f'<text x="{(sml + sw - 24) / 2:.1f}" y="{sh - 10}" class="axis" text-anchor="middle">평일 24시 이후 하차 인원 중앙값(명/일, 로그 눈금)</text>'
          f'<text transform="translate(15,{(sh - smb) / 2 + 8:.1f}) rotate(-90)" class="axis" text-anchor="middle">이용자당 공급 부족(로그 눈금)</text></svg>')
FIGS = "".join(ss)

tt_rows = ""
for t_ in ["간선", "지선", "마을"]:
    a_, b_, c_ = (int(TT.loc[t_].get(k_, 0)) for k_ in ("A1", "A2", "B")); tot_ = a_ + b_ + c_
    tt_rows += f'<tr><td class="b">{t_}</td><td class="n">{a_:,}</td><td class="n">{b_:,}</td><td class="n">{c_:,}</td><td class="n">{tot_:,}</td><td class="n b">{c_ / tot_ * 100:.1f}%</td></tr>'
_tot = TT[["A1", "A2", "B"]].sum()
tt_rows += f'<tr><td class="b">합계(순환 포함)</td><td class="n">{int(_tot["A1"]):,}</td><td class="n">{int(_tot["A2"]):,}</td><td class="n">{int(_tot["B"]):,}</td><td class="n">{int(_tot.sum()):,}</td><td class="n b">{_tot["B"] / _tot.sum() * 100:.1f}%</td></tr>'
A = [ALT.iloc[i] for i in range(3)]
tyt = lambda r: " · ".join(f"{k_} {int(r[k_])}" for k_ in ("간선", "지선", "마을", "순환") if r.get(k_, 0) > 0)
q8_rows = "".join(f'<tr><td class="c">{int(r["qrank"])}</td><td class="b">{r["name"]}{" ★" if r["name"] in ST else ""}</td><td class="n">{r["D_median"]:,.0f}</td><td class="n">{int(r["S1"])}</td><td class="n">{int(r["S2"])}</td><td class="n">{int(r["S3"])}</td><td class="n">{int(r["nB"])}</td><td class="n b">{r["ratio"]:.1f}</td></tr>' for _, r in CQ.head(8).iterrows())
br_rows = "".join(f'<tr><td class="b">{idx}</td><td class="n">{int(BR.loc[idx, "연결"])}</td><td class="n">{int(BR.loc[idx, "15분 이내"])}</td><td class="n">{int(BR.loc[idx, "15분 초과"])}</td><td class="n">{int(BR.loc[idx].sum())}</td></tr>' for idx in BR.index)
Bst = CW["Bst"]; night_has = Bst["S3"].notna().mean(); night_no_link = Bst.loc[Bst["S3"].isna(), "link"].sum()
two_share = CST[CST["name"].isin(ST)]["D_mean"].sum() / CW["tot24"]; q_share = CST[CST["quad"]]["D_mean"].sum() / CW["tot24"]
t8_share = CST[CST["name"].isin(top8)]["D_mean"].sum() / CW["tot24"]; all_share = CST["D_mean"].sum() / CW["tot24"]
mB = TT.loc["마을", "B"] / TT.loc["마을"].sum(); allB = _tot["B"] / _tot.sum()

n_cand = {s_: int((H[H["역"] == s_]["gap"] > 0).sum()) for s_ in ST}
pick_at = lambda s_, th: sum(1 for _, g_, _ in full_order[s_] if g_ >= th)
ex_rows = "".join(f'<tr><td class="c">{k_}</td><td class="n">{M["사당"]["res"]["curve"][k_][1] * 100:.1f}%</td><td class="n">{EX["사당"][k_][0] * 100:.1f}%</td>'
                  f'<td class="n">{M["강변"]["res"]["curve"][k_][1] * 100:.1f}%</td><td class="n">{EX["강변"][k_][0] * 100:.1f}%</td></tr>' for k_ in range(1, 7))

wait_rows = "".join(f'<tr><td class="b">{s_}</td><td class="n">{WAIT[s_]["n_train"]}편</td><td class="c">{hhmm(WAIT[s_]["need"])}</td><td class="n">{WAIT[s_]["one_mean"]:.0f}분</td><td class="n">{WAIT[s_]["one_max"]:.0f}분</td><td class="n">{WAIT[s_]["one_le30"] * 100:.0f}%</td>'
                    f'<td class="c">{hhmm(WAIT[s_]["two_t1"])} + {hhmm(WAIT[s_]["need"])}</td><td class="n">{WAIT[s_]["two_max"]:.0f}분</td><td class="n">{WAIT[s_]["two_mean"]:.0f}분</td></tr>' for s_ in ST)
interval = {s_: WAIT[s_]["need"] - WAIT[s_]["two_t1"] for s_ in ST}
veh2 = lambda r: 1 if 2 * run_est(r) <= interval[r["역"]] else 2
veh1_routes = [r["노선"] for _, r in CH.iterrows() if veh2(r) == 1]
cap_rows = "".join(f'<tr><td class="b">{r["역"]}</td><td class="nw"><b>{r["노선"]}</b> ({r["유형"]}) {r["방면"]} 방면</td><td class="n">{link_sr.get((r["역"], r["노선"]), float("nan")):,.0f}</td><td class="n b">{scn_dir(r):,.0f}</td><td class="n">{r["w"]:.0f}</td><td class="n">{run_est(r):.0f}</td>'
                   f'<td class="c">{veh2(r)}대</td><td class="c nw">{prio_s(r["역"], r["노선"])}</td></tr>' for _, r in CH.iterrows())
fix_rows = ""
for s_ in ST:
    f_ = FIX[s_]
    fix_rows += (f'<tr><td class="b">{s_}</td><td>채택안(승차 가중 선택, {len(f_["sel_b"])}개)</td><td class="nw">{"·".join(i[0] for i in f_["sel_b"])}</td><td class="n b">{pct(f_["b_on_b"])}</td><td class="n">{pct(f_["b_on_a"])}</td></tr>'
                 f'<tr><td class="b"></td><td>하차 가중 최적안({len(f_["sel_a"])}개)</td><td class="nw">{"·".join(i[0] for i in f_["sel_a"])}</td><td class="n">{pct(f_["a_on_b"])}</td><td class="n b">{pct(f_["a_on_a"])}</td></tr>')
fix_gap = {s_: (FIX[s_]["a_best_k"] - FIX[s_]["b_on_a"]) * 100 for s_ in ST}

W, ML, MR = 920, 84, 20
t0, t1 = 22 * 60 + 50, 25 * 60 + 50
pw = W - ML - MR

def TX(t):
    t = min(max(t, t0), t1)
    return ML + (t - t0) / (t1 - t0) * pw

blk = 118
H1 = 30 + blk * 2 + 34
s1 = [f'<svg viewBox="0 0 {W} {H1}" class="chart" role="img" aria-label="역별 지하철 도착 열차와 버스 막차 추정 시각">']
for tm_ in range(23 * 60, t1 + 1, 30):
    s1.append(f'<line x1="{TX(tm_):.1f}" y1="22" x2="{TX(tm_):.1f}" y2="{H1 - 30}" stroke="#ececec"/><text x="{TX(tm_):.1f}" y="{H1 - 14}" class="tick" text-anchor="middle">{hhmm(tm_)}</text>')
s1.append(f'<line x1="{TX(1440):.1f}" y1="16" x2="{TX(1440):.1f}" y2="{H1 - 30}" stroke="#9a9a9a" stroke-width="1.2"/><text x="{TX(1440) + 4:.1f}" y="14" class="ann">자정</text>')
rng = np.random.default_rng(3)
for k, s in enumerate(ST):
    y0 = 30 + blk * k
    s1.append(f'<text x="{ML - 12}" y="{y0 + 16}" class="rowlbl" text-anchor="end">{s}</text>')
    s1.append(f'<text x="{ML - 12}" y="{y0 + 40}" class="tick" text-anchor="end">지하철 도착</text><text x="{ML - 12}" y="{y0 + 84}" class="tick" text-anchor="end">버스 막차</text>')
    s1.append(f'<line x1="{ML}" y1="{y0 + blk - 6}" x2="{W - MR}" y2="{y0 + blk - 6}" stroke="#dcdcdc"/>')
    for a in late[s]:
        s1.append(f'<line x1="{TX(a):.1f}" y1="{y0 + 28}" x2="{TX(a):.1f}" y2="{y0 + 48}" stroke="{"#111" if a >= 1440 else "#b5b5b5"}" stroke-width="1.6"><title>{hhmm(a)} 도착</title></line>')
    need = st_last.loc[s, "last_max"] + TRANSFER
    s1.append(f'<line x1="{TX(need):.1f}" y1="{y0 + 22}" x2="{TX(need):.1f}" y2="{y0 + blk - 10}" stroke="#111" stroke-width="2.2" stroke-dasharray="5 3"/>'
              f'<text x="{TX(need) + 6:.1f}" y="{y0 + 66}" class="ann strong">필요 출발 {hhmm(need)}</text>')
    g = H[H["역"] == s]
    for _, r in g.iterrows():
        yy = y0 + 80 + rng.uniform(-13, 13)
        chosen = pd.notna(r["선택순서"])
        tip = esc(f'{r["노선"]} · {r["방면"]} 방면|추정 막차 {hhmm(r["model"])} · 부족 {r["gap"]:.0f}분|22~23시 승차 {r["w"]:.0f}명/일')
        s1.append(f'<circle class="{"d-ch" if chosen else "d"}" cx="{TX(r["model"]):.1f}" cy="{yy:.1f}" r="{5.5 if chosen else 4}" data-tip="{tip}"/>')
s1.append("</svg>")
FIG1 = "".join(s1)

W2, H2, m2 = 460, 400, 56
a0, a1 = 22 * 60 + 50, 25 * 60 + 50

def P2(x, y):
    return m2 + (x - a0) / (a1 - a0) * (W2 - m2 - 16), H2 - m2 + 10 - (y - a0) / (a1 - a0) * (H2 - m2 - 8)

s2 = [f'<svg viewBox="0 0 {W2} {H2}" class="chart half" role="img" aria-label="추정 막차와 API 예정 막차 비교">']
for tm_ in range(23 * 60, a1 + 1, 30):
    x, _ = P2(tm_, a0); _, y = P2(a0, tm_)
    s2.append(f'<line x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="{H2 - m2 + 10}" stroke="#ececec"/><text x="{x:.1f}" y="{H2 - m2 + 26}" class="tick" text-anchor="middle">{hhmm(tm_)}</text>')
    s2.append(f'<line x1="{m2}" y1="{y:.1f}" x2="{W2 - 16}" y2="{y:.1f}" stroke="#ececec"/><text x="{m2 - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{hhmm(tm_)}</text>')
for off, st in ((0, 'stroke="#111" stroke-width="1.4"'), (10, 'stroke="#9a9a9a" stroke-dasharray="4 3"'), (-10, 'stroke="#9a9a9a" stroke-dasharray="4 3"')):
    x1, y1 = P2(a0 + max(0, -off), a0 + max(0, off)); x2, y2 = P2(a1 - max(0, off), a1 - max(0, -off))
    s2.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {st}/>')
lab = {("5530", "20253"): (8, -8, "start"), (worst["노선"], worst["정류소ARS"]): (8, 4, "start")}
for _, r in v.iterrows():
    x, y = P2(r["api"], r["model"])
    tip = esc(f'{r["역"]} {r["노선"]} · {r["정류소명"]}|추정 {hhmm(r["model"])} · API {hhmm(r["api"])}|차이 {r["err"]:+.0f}분')
    big = (r["노선"], r["정류소ARS"]) in lab
    s2.append(f'<circle class="{"d-ch" if big else "d"}" cx="{x:.1f}" cy="{y:.1f}" r="{5 if big else 4}" data-tip="{tip}"/>')
    if big:
        dx, dy, anc = lab[(r["노선"], r["정류소ARS"])]
        s2.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" class="lbl" text-anchor="{anc}">{r["노선"]} ({r["err"]:+.0f}분)</text>')
s2.append(f'<text x="{(m2 + W2 - 16) / 2:.1f}" y="{H2 - 6}" class="axis" text-anchor="middle">API 예정 막차(2026-09-23 조회)</text>')
s2.append(f'<text transform="translate(13,{(H2 - m2) / 2 + 10:.1f}) rotate(-90)" class="axis" text-anchor="middle">2025 추정 막차(팀 모델)</text></svg>')
FIG2 = "".join(s2)

W3, H3, ml3, mb3 = 920, 330, 56, 46
b0, b1 = 23 * 60 + 30, 25 * 60 + 20

def P3(t, y):
    return ml3 + (t - b0) / (b1 - b0) * (W3 - ml3 - 24), 14 + (1 - y) * (H3 - mb3 - 14)

def conn(s, t):
    return float(np.mean(after0[s] + TRANSFER <= t))

s3 = [f'<svg viewBox="0 0 {W3} {H3}" class="chart" role="img" aria-label="버스 정류소 출발 시각에 따른 자정 이후 지하철 열차 연결률">']
for yv in (0, .25, .5, .75, 1):
    x, y = P3(b0, yv)
    s3.append(f'<line x1="{ml3}" y1="{y:.1f}" x2="{W3 - 24}" y2="{y:.1f}" stroke="#ececec"/><text x="{ml3 - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{int(yv * 100)}%</text>')
for tm_ in range(b0, b1 + 1, 15):
    x, y = P3(tm_, 0)
    s3.append(f'<text x="{x:.1f}" y="{H3 - mb3 + 18}" class="tick" text-anchor="middle">{hhmm(tm_)}</text>')
style = {"사당": 'stroke="#111" stroke-width="2.4"', "강변": 'stroke="#8a8a8a" stroke-width="2.4" stroke-dasharray="6 4"'}
for s in ST:
    ts = np.arange(b0, b1 + 1, 1)
    pts = " ".join(f"{P3(t, conn(s, t))[0]:.1f},{P3(t, conn(s, t))[1]:.1f}" for t in ts)
    s3.append(f'<polyline points="{pts}" fill="none" {style[s]}/>')
marks = []
for _, t in TP.iterrows():
    if t["노선"] not in ("5530", "광진05"):
        continue
    s = t["역"]; row = H[(H["역"] == s) & (H["노선"] == t["노선"]) & (H["정류소명"] == t["정류소명"])].iloc[0]
    right = s == "사당"
    for tt_, lb, dy in ((row["model"], f'{t["노선"]} 현재 {hhmm(row["model"])}', -12), (row["model"] + t["팀 제안 연장"], f'{t["노선"]} +{int(t["팀 제안 연장"])}분 → {pct(conn(s, row["model"] + t["팀 제안 연장"]), 0)}', 18 if right else -12)):
        x, y = P3(tt_, conn(s, tt_))
        s3.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{"#111" if right else "#fff"}" stroke="#111" stroke-width="2"/>'
                  f'<text x="{x + (10 if right else -10):.1f}" y="{y + dy:.1f}" class="ann strong halo" text-anchor="{"start" if right else "end"}">{lb}</text>')
x_, y_ = P3(b1, 0.30)
s3.append(f'<text x="{x_ - 4:.1f}" y="{y_:.1f}" class="ann strong" text-anchor="end">전 열차 연결 시각</text>'
          f'<text x="{x_ - 4:.1f}" y="{y_ + 17:.1f}" class="ann" text-anchor="end">사당 {hhmm(st_last.loc["사당", "last_max"] + TRANSFER)} · 강변 {hhmm(st_last.loc["강변", "last_max"] + TRANSFER)}</text>')
s3.append(f'<text x="{(ml3 + W3 - 24) / 2:.1f}" y="{H3 - 8}" class="axis" text-anchor="middle">버스가 역 정류소를 떠나는 시각(환승 도보 6분 포함 판정)</text>')
s3.append(f'<text transform="translate(13,{(H3 - mb3) / 2 + 8:.1f}) rotate(-90)" class="axis" text-anchor="middle">자정 이후 도착 열차 연결률</text></svg>')
FIG3 = "".join(s3)

cw, chh = 440, 250
s4 = [f'<svg viewBox="0 0 {cw * 2 + 20} {chh}" class="chart" role="img" aria-label="조정 노선 수에 따른 하류 정류소 커버율">']
for k, s in enumerate(ST):
    ox = 10 + k * cw
    c = M[s]["res"]["curve"]; nsel = len(M[s]["res"]["selected"])
    xm = max(len(c) - 1, 8)
    px = lambda v_: ox + 46 + v_ / xm * (cw - 46 - 28)
    py = lambda v_: 26 + (1 - v_) * (chh - 26 - 44)
    s4.append(f'<text x="{ox + 46}" y="16" class="rowlbl">{s}</text>')
    for yv in (0, .5, 1):
        s4.append(f'<line x1="{ox + 46}" y1="{py(yv):.1f}" x2="{ox + cw - 28}" y2="{py(yv):.1f}" stroke="#ececec"/><text x="{ox + 40}" y="{py(yv) + 4:.1f}" class="tick" text-anchor="end">{int(yv * 100)}%</text>')
    for xv in range(0, xm + 1, 2):
        s4.append(f'<text x="{px(xv):.1f}" y="{chh - 26}" class="tick" text-anchor="middle">{xv}</text>')
    s4.append(f'<polyline points="{" ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in c)}" fill="none" stroke="#b5b5b5" stroke-width="2"/>')
    s4.append(f'<polyline points="{" ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in c[:nsel + 1])}" fill="none" stroke="#111" stroke-width="2.6"/>')
    for a, b in c[:nsel + 1]:
        s4.append(f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="3.5" fill="#111"><title>{a}개 노선 · {b * 100:.1f}%</title></circle>')
    s4.append(f'<text x="{px(0) + 6:.1f}" y="{py(c[0][1]) + 16:.1f}" class="ann">조정 전 {c[0][1] * 100:.1f}%</text>')
    s4.append(f'<text x="{px(nsel) + 8:.1f}" y="{py(c[nsel][1]) + 16:.1f}" class="ann strong">{nsel}개 조정 {c[nsel][1] * 100:.1f}%</text>')
    s4.append(f'<text x="{ox + 46 + (cw - 74) / 2:.1f}" y="{chh - 6}" class="axis" text-anchor="middle">조정 노선-방향 수</text>')
s4.append("</svg>")
FIG4 = "".join(s4)

mw, mh = 440, 420
s5 = [f'<svg viewBox="0 0 {mw * 2 + 20} {mh}" class="chart" role="img" aria-label="역별 하류 정류소 커버 지도">']
for k, s in enumerate(ST):
    ox = 10 + k * mw
    p = M[s]["pts"]; la0, lo0 = st_xy.loc[s, "위도"], st_xy.loc[s, "경도"]
    X = (p["lon"].values - lo0) * math.cos(math.radians(la0)) * 111320; Y = (p["lat"].values - la0) * 110540
    lim = np.quantile(np.maximum(np.abs(X), np.abs(Y)), 0.97) * 1.08
    sc = (min(mw, mh) - 50) / 2 / lim
    cx, cy = ox + mw / 2, 24 + (mh - 50) / 2
    PX = lambda x_: cx + np.clip(x_, -lim, lim) * sc
    PY = lambda y_: cy - np.clip(y_, -lim, lim) * sc
    s5.append(f'<rect x="{ox + 4}" y="22" width="{mw - 8}" height="{mh - 46}" fill="#fafafa" stroke="#dcdcdc"/><text x="{ox + 10}" y="16" class="rowlbl">{s}</text>')
    for _, r in CH[CH["역"] == s].iterrows():
        lx = (np.r_[lo0, r["dlon"]] - lo0) * math.cos(math.radians(la0)) * 111320; ly = (np.r_[la0, r["dlat"]] - la0) * 110540
        dash = '' if r["노선"] in common_sel[s] else ' stroke-dasharray="5,3"'
        s5.append(f'<polyline points="{" ".join(f"{PX(a):.1f},{PY(b):.1f}" for a, b in zip(lx, ly))}" fill="none" stroke="#555" stroke-width="1.4"{dash}/>')
        far = int(np.argmax(np.hypot(np.clip(lx, -lim, lim), np.clip(ly, -lim, lim))))
        s5.append(f'<text x="{PX(lx[far]) + 5:.1f}" y="{PY(ly[far]) - 4:.1f}" class="lbl halo">{r["노선"]}</text>')
    order = np.argsort(p["final"].values.astype(int) + p["base"].values.astype(int))
    for i in order:
        cls = "m-base" if p["base"].values[i] else ("m-new" if p["final"].values[i] else "m-none")
        s5.append(f'<circle class="{cls}" cx="{PX(X[i]):.1f}" cy="{PY(Y[i]):.1f}" r="2.6"/>')
    s5.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#fff" stroke="#111" stroke-width="2.5"/><text x="{cx + 11:.1f}" y="{cy + 18:.1f}" class="lbl halo">{s}역</text>')
    km = 2000 if lim > 6000 else 1000
    s5.append(f'<line x1="{ox + 16}" y1="{mh - 34}" x2="{ox + 16 + km * sc:.1f}" y2="{mh - 34}" stroke="#111" stroke-width="2"/><text x="{ox + 16}" y="{mh - 40}" class="tick">{km // 1000}km</text>')
s5.append("</svg>")
FIG5 = "".join(s5)

def trs(rows, cls=None):
    out = ""
    for r in rows:
        out += "<tr>" + "".join(f'<td class="{(cls[i] if cls else "")}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
    return out

ld = last_dir.copy(); ld["t"] = ld["arr"].map(hhmm)
dirname = {"IN": "내선", "OUT": "외선", "UP": "상행", "DOWN": "하행"}
rows_sub = trs([[r["STATION_NM"], f'{r["LINE"]}호선', dirname.get(r["INOUTTAG"], r["INOUTTAG"]), r["t"]] for _, r in ld.sort_values(["STATION_NM", "LINE"], ascending=[False, True]).iterrows()], ["b", "", "", "c"])
jd = H.groupby(["역", "판정"]).size().unstack(fill_value=0)
jcols = ["연결", "경계(모델 오차 범위)", "30분 이내 부족", "30분 초과 부족"]
rows_j = trs([[s, int((H["역"] == s).sum())] + [int(jd.loc[s, c]) if c in jd.columns else 0 for c in jcols] + [f'{gap_med_s[s]:.0f}분', f"{len(after0[s])}편"] for s in ST], ["b", "n", "n", "n", "n", "n", "n", "n"])
rows_ch = ""
for s in ST:
    g = CH[CH["역"] == s]
    for i, (_, r) in enumerate(g.iterrows()):
        rows_ch += ("<tr>" + f'<td class="b">{s if i == 0 else ""}</td><td class="c">{int(r["선택순서"])}</td><td class="nw"><b>{r["노선"]}</b> ({r["유형"]})<br><span class="note">{r["전역분류"]}</span></td><td>{r["정류소명"]}<br>→ {r["방면"]}</td>'
                    f'<td class="c">{hhmm(r["model"])}</td><td class="c b">{hhmm(r["need"])}</td><td class="n">{r["gap"]:.0f}분</td>'
                    f'<td class="n">{pct(r["연결률+30"], 0)}</td><td class="n">{pct(r["연결률+60"], 0)}</td><td class="n">+{r["커버증가"] * 100:.1f}%p</td><td class="n">{r["w"]:.0f}</td><td class="nw">{prio(s, r["노선"])}</td></tr>')
rows_tp = trs([[t["역"], f'<b>{t["노선"]}</b>', f'{t["방면"]} 방면', t["현재(추정)"], f'+{int(t["팀 제안 연장"])}분', t["개선 후"], pct(t["연결률 후"], 0), f'+{int(t["전 열차 연결 필요 연장"])}분', t["API 예정"]] for _, t in TP.iterrows()],
              ["b", "nw", "", "c", "c", "c", "n b", "n b", "c"])
Hs = pd.concat([H[H["역"] == s].sort_values("gap") for s in ST])
rows_all = trs([[r["역"], r["노선"], r["유형"], r["정류소명"], r["방향구간"], r["방면"], r["기점막차"], hhmm(r["model"]), hhmm(r["api"]), f'{r["gap"]:.0f}', pct(r["연결률"], 0), f'{r["w"]:.0f}', f'{r["승차00_월"]:.1f}', r["판정"], r["조치"]] for _, r in Hs.iterrows()],
               ["b", "nw", "", "", "", "", "c", "c", "c", "n", "n", "n", "n", "", ""])
ob = outl.sort_values("act00_월", ascending=False).head(6)
rows_out = trs([[r["역"], r["노선"], r["정류소명"], hhmm(r["model"]), hhmm(r["api"]), f'{r["승차00_월"]:.1f}', f'{r["승차00_일"]:.1f}', "추정이 실제보다 이름(운행 중)" if pd.notna(r["api"]) and r["api"] >= 1440 else ("지연 운행 수준" if r["act00_월"] < 30 else "확인 필요")] for _, r in ob.iterrows()],
                ["b", "nw", "", "c", "c", "n", "n", ""])
sv = {k: srow(k) for k in ["환승 5분", "환승 6분", "환승 7분", "속도계수 0.70", "속도계수 0.85", "속도계수 1.00", "커버 반경 300m", "커버 반경 500m"]}

def cov_txt(x):
    return " · ".join(f'{s} {x["선택 노선"][s]}개({x["커버"][s][0] * 100:.1f}→{x["커버"][s][1] * 100:.1f}%)' for s in ST)

rows_sens = trs([
    ["환승 도보 5 / 6(기준) / 7분", f'연결 {int(sv["환승 5분"]["연결(부족≤0)"])} / {int(sv["환승 6분"]["연결(부족≤0)"])} / {int(sv["환승 7분"]["연결(부족≤0)"])}행,<br>부족 중앙값 {sv["환승 5분"]["부족 중앙값"]:.0f} / {sv["환승 6분"]["부족 중앙값"]:.0f} / {sv["환승 7분"]["부족 중앙값"]:.0f}분'],
    ["심야 속도계수 0.70(API 역산)", f'연결 {int(sv["속도계수 0.70"]["연결(부족≤0)"])}행, 부족 중앙값 {sv["속도계수 0.70"]["부족 중앙값"]:.0f}분<br>{cov_txt(sv["속도계수 0.70"])}'],
    ["심야 속도계수 0.85(기준)", f'연결 {int(sv["속도계수 0.85"]["연결(부족≤0)"])}행, 부족 중앙값 {sv["속도계수 0.85"]["부족 중앙값"]:.0f}분<br>{cov_txt(sv["속도계수 0.85"])}'],
    ["심야 속도계수 1.00(주간 속도)", f'연결 {int(sv["속도계수 1.00"]["연결(부족≤0)"])}행, 부족 중앙값 {sv["속도계수 1.00"]["부족 중앙값"]:.0f}분<br>{cov_txt(sv["속도계수 1.00"])}'],
    ["커버 반경 300m", cov_txt(sv["커버 반경 300m"])],
    ["커버 반경 500m", cov_txt(sv["커버 반경 500m"])],
], ["b", ""])

fonts = "".join('@font-face{font-family:"PretendardEmb";font-weight:%d;font-style:normal;font-display:swap;src:url(data:font/woff2;base64,%s) format("woff2");}'
                % (w_, base64.b64encode(open(FONT + fn, "rb").read()).decode()) for w_, fn in ((400, "Pretendard-Regular.woff2"), (600, "Pretendard-SemiBold.woff2"), (800, "Pretendard-ExtraBold.woff2")))
CSS = """
:root{--ink:#111;--ink2:#444;--ink3:#777;--line:#d9d9d9;--line2:#ececec;--panel:#f5f5f5}
*{box-sizing:border-box}
body{margin:0;background:#e9e9e9;color:var(--ink);font-family:"PretendardEmb","Pretendard",-apple-system,"Malgun Gothic",sans-serif;font-size:14.5px;line-height:1.65;word-break:keep-all;-webkit-font-smoothing:antialiased}
.page{max-width:1020px;margin:32px auto;background:#fff;padding:48px 56px 56px;border:1px solid var(--line)}
@media (max-width:700px){.page{padding:28px 18px;margin:0}}
.meta{font-size:12.5px;color:var(--ink3)}
h1{font-size:26px;font-weight:800;letter-spacing:-.02em;line-height:1.35;margin:10px 0 6px}
.lead{font-size:15.5px;color:var(--ink2);margin:0 0 24px}
h2{font-size:18.5px;font-weight:800;margin:44px 0 6px;padding-top:18px;border-top:2px solid var(--ink)}
h2 .no{display:inline-block;min-width:26px;color:var(--ink3);font-weight:600}
h2 .form{font-size:12px;font-weight:600;color:var(--ink3);border:1px solid var(--line);padding:1px 7px;margin-left:8px;vertical-align:2px}
h3{font-size:15px;font-weight:800;margin:26px 0 8px}
.sec-lead{color:var(--ink2);margin:4px 0 14px}
.summary{background:var(--panel);border-left:4px solid var(--ink);padding:20px 24px}
.summary h2{border:0;margin:0 0 12px;padding:0;font-size:16px}
.summary ul{margin:12px 0 0;padding-left:18px}.summary li{margin:6px 0}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
@media (max-width:700px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{border:1px solid var(--line);background:#fff;padding:14px 16px}
.kpi .v{font-size:23px;font-weight:800;line-height:1.15}.kpi .k{font-size:12.5px;color:var(--ink3);margin-top:6px}
.pick{margin:16px 0 0;padding:12px 16px;border:1.5px solid var(--ink);background:#fff;font-weight:800}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:12px 0}
@media (max-width:700px){.flow{grid-template-columns:1fr}}
.flow div{border:1px solid var(--line);padding:10px 12px;font-size:13px}.flow b{display:block;font-size:12px;color:var(--ink3)}
.eq{border:1px solid var(--line);background:#fafafa;padding:10px 14px;margin:10px 0;font-weight:600}
.tw{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-weight:800;background:#f0f0f0;border-top:1.5px solid var(--ink);border-bottom:1px solid var(--line);padding:7px 9px;white-space:nowrap}
td{border-bottom:1px solid var(--line2);padding:7px 9px;vertical-align:top}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}td.c{text-align:center;white-space:nowrap}.b{font-weight:800}.nw{white-space:nowrap}
ul{padding-left:18px}li{margin:4px 0}ol{padding-left:20px}ol li{margin:6px 0}
.tag{display:inline-block;font-size:11.5px;font-weight:800;border:1px solid var(--ink);padding:0 6px;line-height:18px}
.tag.m{border-color:var(--ink3);color:var(--ink3)}
figure{margin:18px 0 8px}figcaption{font-size:13px;font-weight:800;margin-bottom:4px}
.chart{width:100%;height:auto;display:block}.chart.half{max-width:520px}
.chart .tick{font-size:11px;fill:#777}.chart .axis{font-size:12px;fill:#444}.chart .ann{font-size:11.5px;fill:#444}.chart .ann.strong{font-weight:800;fill:#111}
.chart .halo{paint-order:stroke;stroke:#fff;stroke-width:3.5px;stroke-linejoin:round}
.chart .big{font-size:14.5px}
.chart .rowlbl{font-size:13.5px;font-weight:800;fill:#111}.chart .lbl{font-size:12px;font-weight:800;fill:#111}
.d{fill:#fff;stroke:#8a8a8a;stroke-width:1.3}.d-ch{fill:#111;stroke:#fff;stroke-width:1.5}
.m-none{fill:#fff;stroke:#9a9a9a;stroke-width:1}.m-base{fill:#b5b5b5;stroke:#fff;stroke-width:.6}.m-new{fill:#111;stroke:#fff;stroke-width:.6}
.chart circle[data-tip]{cursor:pointer}.chart circle[data-tip]:hover{stroke:#111;stroke-width:2.2}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin:6px 0 0}
.legend i{display:inline-block;width:11px;height:11px;margin-right:6px;vertical-align:-1px}
.two{display:grid;grid-template-columns:minmax(0,520px) 1fr;gap:22px;align-items:start}
@media (max-width:820px){.two{grid-template-columns:1fr}}
#tip{position:fixed;pointer-events:none;background:#111;color:#fff;font-size:12.5px;padding:8px 10px;line-height:1.5;display:none;z-index:9}
.note{font-size:12.5px;color:var(--ink3)}details{margin:10px 0}summary{cursor:pointer;font-weight:800}
.src li{font-size:13px;color:var(--ink2)}a{color:var(--ink)}
.foot{margin-top:36px;font-size:12px;color:var(--ink3);border-top:1px solid var(--line);padding-top:12px}
@media print{body{background:#fff}.page{border:0;margin:0;max-width:none}details{display:block}}
"""

ch_names = {s: "·".join(CH[CH["역"] == s]["노선"]) for s in ST}
ch1 = {s: "·".join(r_ for r_ in CH[CH["역"] == s]["노선"] if r_ in common_sel[s]) for s in ST}
ch2 = {s: "·".join(r_ for r_ in CH[CH["역"] == s]["노선"] if r_ not in common_sel[s]) or "없음" for s in ST}
n_dep = n_ch - n_common
L = lambda t, u: f'<a href="{u}">{t}</a>'
BODY = f"""
<div class="meta">데이터로(路) · AI와 함께하는 교통문제 해결을 위한 데이터 분석 공모전 · 작성 2026-10-01 · 분석 기준 2025년</div>
<h1>서울 지하철–버스 막차 환승 단절 진단과 연계 막차 우선 조정안</h1>
<p class="lead">사당·강변역 사례 분석,<br>검증 과정과 결과·시각화를 포함한 최종 산출물</p>

<section class="summary">
<h2>요약</h2>
<div class="kpis">
<div class="kpi"><div class="v">{n_q}/{n_st}곳</div><div class="k">심야 수요 상위이면서 버스 공급 부족 상위인 역</div></div>
<div class="kpi"><div class="v">{n_conn_u}/{n_u}</div><div class="k">사당·강변에서 지하철 최종 막차와 연결되는 버스 노선-방향(정류소 중복 제외)</div></div>
<div class="kpi"><div class="v">{n_common}+{n_dep}개</div><div class="k">연계 막차 조정 노선-방향(1순위 공통 {n_common} · 2순위 가정 의존 {n_dep})</div></div>
<div class="kpi"><div class="v">{cov["사당"][0] * 100:.0f}→{cov["사당"][1] * 100:.0f}% · {cov["강변"][0] * 100:.0f}→{cov["강변"][1] * 100:.0f}%</div><div class="k">하류 정류소 커버율, 승차 가중(사당 · 강변)</div></div>
</div>
<ul>
<li><b>서울 전체:</b> {n_st}개 역 주변 귀가방향 버스 {CW["home_n"]:,}개 노선-정류장 중 {allB * 100:.1f}%는 자정 이후 이용 기록이 없음(마을버스 {mB * 100:.1f}%).<br>기점 막차 기반 추정과 {AGREE["B"] * 100:.1f}% 일치하나 실제 운행 확인은 아니며,<br>정책 대안 3종의 대상은 ① {int(A[0]["노선"])}개 노선, ② 기준 시나리오에서 {int(A[1]["노선"])}개(기준선에 따라 {BS[0][1]}~{BS[3][1]}개), ③ {int(A[2]["노선"])}개임</li>
<li><b>역 도출:</b> 수요 상위 25%이면서 이용자당 공급 부족 상위 25%인 역 {n_q}곳 중,<br>공급 부족 1위 사당과 2위 강변을 심층 분석 대상으로 정함</li>
<li><b>진단:</b> 두 역의 귀가방향 버스 {n_u}개 노선-방향(정류소 중복 포함 {n_H}행) 중 지하철 최종 막차 승객이 탈 수 있는 것은 {n_conn_u}개(모두 사당)이며,<br>버스 막차가 필요한 시각보다 중앙값 {gap_med_u:.0f}분 먼저 끊김</li>
<li><b>처방:</b> 하류 정류소가 겹치지 않도록 고른 {n_ch}개 노선-방향에 연계 막차를 1회씩 추가하면,<br>하류 정류소 커버율(승차 가중, 최종 막차 승객 기준)이 사당 {pct(cov["사당"][0])}→{pct(cov["사당"][1])}, 강변 {pct(cov["강변"][0])}→{pct(cov["강변"][1])}로 상승함.<br>탐욕 선택 결과는 노선 1~6개 범위의 조합 전수 비교에서 최대 커버율과 일치함</li>
<li><b>우선순위:</b> {n_ch}개 중 {n_common}개는 승차·하차 두 가중치에서 공통으로 선택된 1순위이고, {n_dep}개는 승차 가중 가정에 의존하는 2순위임.<br>채택안을 고정한 채 하차 가중으로 다시 계산해도 커버율은 사당 {pct(FIX["사당"]["b_on_a"])}, 강변 {pct(FIX["강변"]["b_on_a"])}로 유지됨(같은 수의 하차 가중 최적안 대비 −{fix_gap["사당"]:.1f}·−{fix_gap["강변"]:.1f}%p)</li>
<li><b>대기시간:</b> 연계 막차를 1회만 두면 자정 이후 열차 승객의 최대 대기가 사당 {WAIT["사당"]["one_max"]:.0f}분·강변 {WAIT["강변"]["one_max"]:.0f}분이며,<br>{hhmm(WAIT["사당"]["two_t1"])}·{hhmm(WAIT["강변"]["two_t1"])}에 1회를 더 두면 최대 {WAIT["사당"]["two_max"]:.0f}분·{WAIT["강변"]["two_max"]:.0f}분으로 줄어듦</li>
<li><b>검증 ①:</b> 팀의 2025년 막차 추정 모델은 2026년 API 예정 막차와 평균 {err_mae:.1f}분 차이(±20분 이내 {pct(in20, 0)})로 자료 간 일관성은 확인함.<br>다만 2025년 실제 운행 시각과의 정확도는 미확인이므로, 부족 ±10분 이내는 '경계'로 분류함</li>
<li><b>검증 ②:</b> 5530번은 추정 {hhmm(r5530["model"])}이나 실제 예정 {hhmm(r5530["api"])}로 자정 이후에도 운행 중임.<br>'자정 전 통과 추정인데 00시 승차 발생'은 공급 단절이 아니라 모델 오차이므로, 근거를 지하철 막차와의 시간 차로 교체함</li>
<li><b>검증 ③:</b> 팀 제안 '기점 막차 +30분'은 자정 이후 도착 열차의 {pct(TP["연결률 후"].min(), 0)}~{pct(TP["연결률 후"].max(), 0)}만 연결함.<br>전 열차 연결에는 +{int(TP["전 열차 연결 필요 연장"].min())}~{int(TP["전 열차 연결 필요 연장"].max())}분이 필요해, 연장보다 '연계 막차 1회 추가'가 적합함</li>
<li><b>우선순위 조정:</b> 5530번은 같은 군포 방면 540번(추정 막차 {hhmm(r540["model"])})이 목적지의 {pct(r5530["행커버"], 0)}를 이미 연결해 사당 {rank5530}순위임.<br>강변은 팀 제안(강동01·광진03·광진05)과 본 분석 상위 3개가 일치함</li>
</ul>
<div class="pick">1순위(두 가중치 공통 {n_common}개) &nbsp;|&nbsp; 사당: {ch1["사당"]} &nbsp;/&nbsp; 강변: {ch1["강변"]}<br>2순위(승차 가중 의존 {n_dep}개) &nbsp;|&nbsp; 사당: {ch2["사당"]} &nbsp;/&nbsp; 강변: {ch2["강변"]}</div>
</section>

<h2><span class="no">1</span>주제 <span class="form">양식 항목</span></h2>
<p><b>서울 지하철–버스 막차 환승 단절 진단과 연계 막차 우선 조정안: 사당·강변역 사례</b></p>
<ul>
<li>산출물 형태: 역 × 노선 × 분 단위 처방 목록({n_ch}개 노선-방향, 1순위 {n_common} + 2순위 {n_dep})</li>
<li>핵심 질문: 어느 역의 어떤 버스에 막차를 더하면,<br>가장 적은 운행으로 가장 많은 귀갓길을 이을 수 있는가</li>
</ul>

<h2><span class="no">2</span>분석 배경 및 목적 <span class="form">양식 항목</span></h2>
<ul>
<li><b>문제 상황:</b> 평일 지하철은 01시 전후까지 운행하나,<br>역에서 주거지로 가는 시내·마을버스는 대부분 자정 전후에 끝남</li>
<li><b>정책 공백:</b> 서울시–마을버스조합 추가 합의문(2025-12-18)은 '첫·막차 및 배차간격 준수'만 규정하며,<br>지하철 막차와의 연계 기준은 없음</li>
<li><b>대상 역 선정:</b> 서울교통공사 {n_st}개 역의 심야 수요 대비 버스 공급 격차 분석에서<br>이용자당 공급 부족 1위 사당, 2위 강변을 선정함(6-3)</li>
<li><b>두 역의 성격:</b> 사당은 2·4호선 환승 거점(평일 24시 이후 하차 중앙값 {f0(DMED["사당"])}명),<br>강변은 마을버스 의존형 주거 연결 역({f1(DMED["강변"])}명)으로 유형이 달라 비교 사례가 됨</li>
<li><b>목적:</b> 새 노선을 만들지 않고,<br>기존 노선의 막차 시각 조정만으로 이을 수 있는 귀갓길을 찾음</li>
</ul>

<h2><span class="no">3</span>활용 데이터 <span class="form">양식 항목</span></h2>
<p class="sec-lead">주 분석은 2025년 자료만 사용하며,<br>2026년 API 값은 추정 모델 검증에만 사용함</p>
<div class="tw"><table><tr><th>활용 데이터</th><th>제공기관</th><th>출처 플랫폼</th><th>기준</th><th>용도 · 선정 이유</th></tr>
<tr><td>{L("서울 도시철도 열차운행시각표", "https://www.data.go.kr/data/15098251/fileData.do")}</td><td>서울교통공사</td><td>공공데이터포털</td><td>2025-09-30</td><td>평일 방향별 막차 도착, 23시 이후 도착 열차 목록</td></tr>
<tr><td>{L("역별 일별 시간대별 승하차인원", "https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do")}</td><td>서울교통공사</td><td>서울 열린데이터광장</td><td>2025년</td><td>역별 24시 이후 하차(역 선정, 효과 규모)</td></tr>
<tr><td>{L("버스노선별 정류장별 시간대별 승하차 인원", "https://data.seoul.go.kr/dataList/OA-12913/S/1/datasetView.do")}</td><td>서울특별시</td><td>서울 열린데이터광장</td><td>2025년 1~12월</td><td>22~23시 승차(가중치), 00시 승하차(실측 대조)</td></tr>
<tr><td>{L("버스노선별 경유정류소 목록정보", "https://data.seoul.go.kr/dataList/OA-15261/S/1/datasetView.do")}</td><td>서울특별시</td><td>서울 열린데이터광장</td><td>2025-12-09</td><td>정류소 순번 → 방향·하류 정류소</td></tr>
<tr><td>{L("버스노선 기본정보", "https://data.seoul.go.kr/dataList/OA-15262/S/1/datasetView.do")}</td><td>서울특별시</td><td>서울 열린데이터광장</td><td>2025-12-09</td><td>기점 막차, 총 운행시간, 기점·종점명</td></tr>
<tr><td>{L("버스정류소 위치정보", "https://data.seoul.go.kr/dataList/OA-15067/S/1/datasetView.do")}</td><td>서울특별시</td><td>서울 열린데이터광장</td><td>2025-12-09</td><td>역 반경 300m 정류소 추출</td></tr>
<tr><td>{L("1~8호선 역사 좌표", "https://www.data.go.kr/data/15099316/fileData.do")}</td><td>서울교통공사</td><td>공공데이터포털</td><td>2025-08-14</td><td>역 중심 좌표</td></tr>
<tr><td>{L("정류소정보조회 서비스(API)", "https://www.data.go.kr/data/15000303/openapi.do")}</td><td>서울특별시</td><td>공공데이터포털</td><td>2026-09-23 조회</td><td><b>검증 전용</b>: 정류소별 예정 막차와 추정값 비교</td></tr>
</table></div>
<p class="note">모든 자료는 공공누리 출처표시 조건의 공개 자료이며, 자동수집 없이 공식 다운로드·API로 확보함</p>

<h2><span class="no">4</span>분석과정 및 방법 <span class="form">양식 항목</span></h2>
<div class="flow">
<div><b>1단계</b>서울 전체 진단<br>이용 기록 등급 · 대안 3종 규모</div>
<div><b>2단계</b>역 선정<br>수요 대비 공급 격차</div>
<div><b>3단계</b>방향 구분 · 막차 추정<br>정류소 순번 · 기점 막차</div>
<div><b>4단계</b>연계 판정<br>지하철 막차 + 환승</div>
<div><b>5단계</b>우선 조정안<br>탐욕 선택 + 조합 전수 비교</div>
</div>
<h3>4-0. 서울 전체 진단과 역 선정 기준</h3>
<div class="tw"><table><tr><th>항목</th><th>기준</th></tr>
<tr><td class="b nw">이용 기록 등급</td><td>역 반경 300m 정류장의 노선별 2025년 월별 승하차로 판정하고, 12개월 최빈값을 사용함(월별 일치율 {CW["stab"] * 100:.1f}%)<br>01시 승하차가 있으면 '01시 이후 이용 기록', 00시만 있으면 '00시대 이용 기록', 둘 다 없으면 '자정 이후 이용 기록 없음'<br>이용 기록이 없다고 운행이 끝났다고 단정할 수 없으므로, 기점 막차 + 주행시간 추정(4-2 모델)과의 일관성을 확인함(6-1, 실제 시간표 대조는 아님)</td></tr>
<tr><td class="b nw">해당 노선 몫</td><td>역의 평일 24시 이후 하차 × 그 노선의 역 귀가방향 23시 승차 비중</td></tr>
<tr><td class="b nw">대안 구분</td><td>① 00시대 이용 기록 노선 / ② 자정 이후 이용 기록 없음 + 노선 몫이 기준선 이상 / ③ 자정 이후 이용 기록 없음 + 기준선 미만<br>기준선 {CW["bench"]:.1f}명 = 00시대에 운행 중인 일반노선 {CW["n_late"]}개의 노선 전체 00시대 1일 승차 중앙값.<br>비교 대상(역 배분 몫)과 범위가 다른 가정값이므로 6-2에서 기준선 민감도를 제시함</td></tr>
<tr><td class="b nw">역 선정 지표</td><td>수요 = 평일 24시 이후 하차 중앙값, 공급 = 00시 이후 이용 기록이 있는 일반노선 수 + 올빼미 노선 수<br>이용자당 공급 부족 = 수요 ÷ (공급 + 1)</td></tr>
<tr><td class="b nw">선정 규칙</td><td>수요 상위 25%이면서 이용자당 공급 부족 상위 25%인 역(4분면)<br>수요–공급 격차로 취약지역을 정의한 Currie(2010), Jiao &amp; Dillivan(2013)과 시간대별 측정을 제안한 Fransen 외(2015)를 따름</td></tr>
</table></div>
<h3>4-1. 분석 단위(사당·강변 심층)</h3>
<ul>
<li>역 반경 300m 정류소를 지나는 (정류소, 노선, 방향) {n_R}행(사당 {(R["역"] == "사당").sum()}, 강변 {(R["역"] == "강변").sum()})</li>
<li>귀가방향: 2025년 23시 승차가 일평균 1명 이상인 정류소-노선 → 분석 대상 {n_H}행 = 고유 노선-방향 {n_u}개<br>(같은 노선-방향이 역 주변 두 정류소에서 나타나면 2행으로 세고, 고유 집계는 더 늦게 지나는 정류소 기준)</li>
<li>방향: 노선기본정보의 종점명으로 회차점을 찾아 기점→종점, 종점→기점으로 나눔.<br>역 정류소가 회차점이면 되돌아가는 구간을 귀가 구간으로 봄</li>
</ul>
<h3>4-2. 버스 막차 통과 시각 추정(팀 모델)</h3>
<div class="eq">정류소 막차 통과 시각 = 기점 막차 + (정류소 순번 ÷ 총 정류소 수) × 총 운행시간 × 0.85</div>
<ul>
<li>0.85는 심야 주행이 주간보다 빠르다는 가정이며, 운행시간이 없는 노선은 정류소당 1.45분을 적용함</li>
<li>두 값은 근거 문헌이 없는 가정이므로, 5절에서 API로 검증하고 민감도를 제시함</li>
</ul>
<h3>4-3. 연계 판정</h3>
<div class="eq">부족 시간 = (역의 지하철 최종 막차 도착 + 환승 도보 6분) − 버스 막차 통과 시각</div>
<ul>
<li>환승 도보 6분은 팀 가정(5~7분)의 중간값이며, 현장 실측으로 교체 예정</li>
<li>모델 오차를 고려해 부족 시간 ±10분 이내는 '경계'로 따로 분류함</li>
<li>연결률: 자정 이후 도착하는 평일 열차 중, 환승 후 그 버스를 탈 수 있는 열차의 비율</li>
</ul>
<h3>4-4. 우선 조정안 선택(탐욕 선택)</h3>
<ul>
<li>방법: 최대 커버 모형(Church &amp; ReVelle, 1974)의 탐욕 해법</li>
<li>수요점: 귀가방향 노선-방향의 하류 정류소, 가중치는 역 정류소의 2025년 22~23시 승차 일평균을 하류 정류소 수로 나눠 배분<br>이 가중치는 역에서 그 방향으로 타는 사람의 규모이지 목적지별 수요가 아니므로, 결과는 '승차 가중 정류소 커버율'로 부름.<br>하류 정류소의 22~00시 하차를 가중치로 쓴 결과를 6-8에서 비교함</li>
<li>중복 처리: 같은 노선-방향이 두 정류소에 있으면 각 정류소의 승차를 따로 가중치로 쓰고(승객이 다름), 하류 정류소가 겹치면 합집합으로 커버함</li>
<li>기준: 커버는 지하철 최종 막차 승객 기준이며, 필요 출발 시각 이후에도 운행하는 노선만 기존 서비스로 인정함</li>
<li>커버: 수요점 400m 안에 막차 이후 이용 가능한 노선의 정류소가 있으면 커버(TCQSM의 버스 정류소 400m 기준)</li>
<li>기존 서비스: 이미 연결되는 노선과 역 300m를 지나는 올빼미버스(사당 N15)</li>
<li>선택: 남은 하류 정류소를 가장 많이 잇는 노선부터 추가하고, 증가가 5%p 미만이면 중단함(운영 효율 가정: 노선 1개 추가로 가중 수요의 1/20도 못 잇는 경우 제외)</li>
<li>검증: 탐욕 선택은 최적을 보장하지 않으므로, 후보 조합을 전수 비교해 같은 노선 수에서 얻을 수 있는 최대 커버율과 대조함(6-8)</li>
</ul>

<h2><span class="no">5</span>검증 과정</h2>
<p class="sec-lead">팀 산출물을 원자료로 다시 계산하고,<br>추정 모델·이상치 해석·정책 제안을 차례로 점검함</p>
<div class="tw"><table><tr><th>검증 항목</th><th>방법</th><th>결과</th><th>조치</th></tr>
<tr><td class="b">① 산출값 재현</td><td>팀 산식을 원자료에 다시 적용</td><td>{repro_n}행 중 {repro_ok}행이 1분 이내 일치.<br>분석 대상 {n_H}행은 전부 일치</td><td>팀 산출값을 그대로 채택</td></tr>
<tr><td class="b">② 추정 모델 정확도</td><td>API 예정 막차({len(v)}행)와 비교<br>(모든 시각은 표시값과 같은 분 단위 내림값으로 계산)</td><td>평균 {err_mean:+.1f}분, 평균 절대오차 {err_mae:.1f}분.<br>±10분 {pct(in10)}, ±20분 {pct(in20)}</td><td>자료 간 일관성 확인(2026년 예정값 대조).<br>2025년 실제 운행 정확도는 미확인이라 부족 ±10분 이내는 '경계'로 분류</td></tr>
<tr><td class="b">③ 이상치 해석</td><td>'자정 전 통과 추정 + 00시 승하차' 행을 API와 대조</td><td>{len(outl)}행 중 월 30명 이상은 {len(outl_big)}행(모두 5530번).<br>API상 자정 이후 운행 {outl_api_after}행</td><td>'공급 단절 증거'에서 '모델이 이르게 잡은 사례'로 재분류</td></tr>
<tr><td class="b">④ 팀 제안안 효과</td><td>+25~30분 연장 시 연결되는 열차 비율 계산</td><td>자정 이후 열차의 {pct(TP["연결률 후"].min(), 0)}~{pct(TP["연결률 후"].max(), 0)} 연결</td><td>'100% 흡수' 표현 삭제, 연계 막차 1회 추가로 보완</td></tr>
<tr><td class="b">⑤ 방향 구분</td><td>종점명 기반 회차점 재판정</td><td>종점이 그 역인 노선(동작09, 2224 등)의 귀가 구간이 빠져 있던 문제 수정</td><td>수정 규칙 적용</td></tr>
<tr><td class="b">⑥ 자료 판본</td><td>2025·2026판 비교</td><td>정류소 위치: 두 역 300m 목록 동일.<br>시각표: 6개 방향 중 5개 동일(사당 4호선 하행만 4분 차)</td><td>2025년판으로 통일</td></tr>
</table></div>
<div class="two">
<figure><figcaption>그림 1. 추정 막차와 API 예정 막차 비교({len(v)}행)</figcaption>{FIG2}
<div class="legend"><span><i style="background:#111;height:2px"></i>일치선</span><span><i style="border-top:2px dashed #9a9a9a;height:0"></i>±10분</span></div></figure>
<div>
<h3>모델 검증에서 확인한 점</h3>
<ul>
<li>추정값이 API보다 평균 {err_mean:.0f}분 늦음.<br>2026년 예정값 기준으로는 버스가 더 일찍 끊기는 쪽이므로 부족 시간이 과소 추정됐을 수 있으나, 2025년 실제 운행과의 차이는 미확인임</li>
<li>가장 큰 차이는 {worst["역"]} {worst["노선"]}번({worst["err"]:+.0f}분)임</li>
<li>5530번({hhmm(r5530["model"])} 추정, {hhmm(r5530["api"])} 예정)처럼 추정이 이른 경우는 운행시간 정보가 없는 노선에서 나타남</li>
<li>API는 2026-09-23 값이라 2025년 시각표와 같다는 보장은 없음.<br>검증에만 쓰고 결과 수치에는 넣지 않음</li>
</ul>
</div></div>
<h3>이상치 재해석: 자정 전 통과로 추정됐으나 00시 승하차가 있는 행(상위 6개)</h3>
<div class="tw"><table><tr><th>역</th><th>노선</th><th>정류소</th><th>추정 막차</th><th>API 예정</th><th>00시 승차(월평균)</th><th>00시 승차(일평균)</th><th>해석</th></tr>{rows_out}</table></div>
<ul>
<li>5530번의 00시 승차 월 {r5530["승차00_월"]:.1f}명은 하루 {r5530["승차00_일"]:.1f}명으로, 예정 막차({hhmm(r5530["api"])})의 정상 승차 규모임</li>
<li>API 예정 막차가 자정 이후인 5530·5524번은 추정이 실제보다 이른 경우이며,<br>그 밖의 행은 월 30명 미만으로 막차가 몇 분 늦게 지나가 00시로 기록된 수준임</li>
</ul>

<h2><span class="no">6</span>분석 내용 및 결과 <span class="form">양식 항목</span></h2>
<h3>6-1. 서울 전체 진단: 귀가방향 버스의 자정 이후 이용 기록</h3>
<div class="tw"><table><tr><th>유형</th><th>01시 이후 이용 기록</th><th>00시대 이용 기록</th><th>자정 이후 이용 기록 없음</th><th>합계</th><th>기록 없음 비중</th></tr>{tt_rows}</table></div>
<ul>
<li>대상: 서울교통공사 {n_st}개 역 반경 300m, 귀가방향 {CW["home_n"]:,}개 노선-정류장(2025년 12개월 최빈 등급)</li>
<li>자정 이후 이용 기록이 없는 비중은 마을버스에서 가장 높아, 역에서 먼 주거지 구간의 심야 이용 기록이 먼저 사라짐</li>
<li>추정 간 교차검증: 기점 막차 + 주행시간 추정(4-2 모델)으로 다시 판정하면 '이용 기록 없음' {nB_chk:,}쌍 중 {AGREE["B"] * 100:.1f}%가 자정 전 통과 추정과 일치함('01시 이후' {AGREE["A1"] * 100:.0f}%, '00시대' {AGREE["A2"] * 100:.0f}% 일치).<br>이는 이용 기록과 모델 추정이라는 서로 다른 두 추정 간 일관성이며, 실제 정류소 시간표와의 대조가 아니므로 실제 운행 정확도는 미확인임.<br>나머지 {100 - AGREE["B"] * 100:.1f}%는 버스가 다니지만 이용 기록이 없었을 수 있어, 이 등급은 '운행 종료'가 아니라 '이용 기록 없음'으로 읽어야 함</li>
<li>교차검증 대상: '이용 기록 없음' {nB_home:,}쌍 중 {nB_ex}쌍은 제외됨({xB_txt}).<br>제외 쌍은 기점 막차 자료가 없어 추정 자체가 불가한 경우임</li>
<li>1월·12월은 연말 한시 연장 운행으로 '01시 이후 이용 기록' 비중이 28~29%로 높아지므로, 단월이 아닌 12개월 최빈값을 사용함</li>
</ul>
<h3>6-2. 정책 대안 3종의 적용 규모(서울 전체)</h3>
<figure><figcaption>그림 2. 정책 대안 3종의 대상 노선 수와 해당 노선 몫 하차 인원</figcaption>{FIGA}</figure>
<div class="tw"><table><tr><th>대안</th><th>대상</th><th>노선 수(유형)</th><th>역 수</th><th>해당 노선 몫<br>24시 이후 하차</th><th>추가 자원</th><th>운영 가능성</th></tr>
<tr><td class="b">① 5~15분 막차 조정</td><td>00시대 이용 기록 노선</td><td class="nw"><b>{int(A[0]["노선"])}개</b><br>{tyt(A[0])}</td><td class="n">{int(A[0]["역"])}</td><td class="n b">{A[0]["link"]:,.0f}명/평일</td><td>없음(종료 시각만 연장)</td><td><b>상</b><br>대상 확정에 분 단위 막차 확인 필요</td></tr>
<tr><td class="b">② 막차 1회 추가</td><td>자정 이후 이용 기록 없음,<br>노선 몫 기준선 이상</td><td class="nw"><b>{int(A[1]["노선"])}개</b>(기준 시나리오)<br>{tyt(A[1])}<br><span class="note">기준선 {BS[0][0]:.0f}~{BS[3][0]:.0f}명이면 {BS[0][1]}~{BS[3][1]}개</span></td><td class="n">{int(A[1]["역"])}</td><td class="n b">{A[1]["link"]:,.0f}명/평일</td><td>노선당 심야 1회전</td><td><b>중</b><br>연말 한시 연장 선례, 마을버스는 기사 부족</td></tr>
<tr><td class="b">③ 대체수단 연계</td><td>자정 이후 이용 기록 없음,<br>노선 몫 기준선 미만</td><td class="nw"><b>{int(A[2]["노선"])}개</b><br>{tyt(A[2])}</td><td class="n">{int(A[2]["역"])}</td><td class="n b">{A[2]["link"]:,.0f}명/평일</td><td>올빼미 환승 안내 ~ 심야 DRT</td><td><b>중장기</b><br>해당 역의 {night_has * 100:.1f}%는 올빼미 정류소 보유</td></tr>
</table></div>
<ul>
<li>세 대안은 대상 노선이 겹치지 않아, 택일이 아닌 조합으로 운영할 수 있음</li>
<li>대안①의 {A[0]["link"]:,.0f}명은 이미 버스로 연결되는 승객을 포함한 상한값이며,<br>실제 조정 대상은 분 단위 막차 확인 후 줄어듦(6-4)</li>
<li>대안② {int(A[1]["노선"])}개 노선의 {int(A[1]["마을"]) / int(A[1]["노선"]) * 100:.0f}%가 마을버스로, 운영 여력이 약한 유형에 효과가 집중됨</li>
<li>자정 이후 이용 기록이 없는 노선이 있는 역 중 올빼미 정류소가 없는 역의 연계 수요는 {night_no_link:,.0f}명/평일임</li>
<li>기준선 민감도: ②·③을 가르는 기준선을 {BS[0][0]:.0f} / {BS[1][0]:.0f}(기준) / {BS[2][0]:.0f} / {BS[3][0]:.0f}명으로 바꾸면,<br>대안② 노선 수는 {BS[0][1]} / {BS[1][1]} / {BS[2][1]} / {BS[3][1]}개(노선 몫 합 {BS[0][2]:,.0f} / {BS[1][2]:,.0f} / {BS[2][2]:,.0f} / {BS[3][2]:,.0f}명)로 바뀜.<br>기준선은 역 배분 몫과 비교 범위가 다른 가정값이므로, 정책 대상 규모는 범위로 읽어야 함</li>
</ul>
<h3>6-3. 사당·강변 도출 과정</h3>
<div class="tw"><table><tr><th>단계</th><th>기준</th><th>남는 역</th><th>24시 이후<br>하차 비중</th></tr>
<tr><td class="b nw">① 분석 대상</td><td>서울교통공사 역 중 반경 300m에 귀가방향 버스 노선이 있는 역</td><td class="n">{n_st}곳</td><td class="n">{all_share * 100:.1f}%</td></tr>
<tr><td class="b nw">② 우선 검토 영역</td><td>수요 상위 25%({qD:.0f}명 이상)이면서 이용자당 공급 부족 상위 25%({qR:.1f} 이상)</td><td class="n">{n_q}곳</td><td class="n">{q_share * 100:.1f}%</td></tr>
<tr><td class="b nw">③ 상위 8곳</td><td>우선 검토 영역 중 이용자당 공급 부족이 큰 순서</td><td class="n">8곳</td><td class="n">{t8_share * 100:.1f}%</td></tr>
<tr><td class="b nw">④ 심층 분석</td><td>공급 부족 1·2위이며 유형이 다른 두 역(환승 거점형 사당, 마을버스 의존형 강변)</td><td class="n">2곳</td><td class="n">{two_share * 100:.1f}%</td></tr>
</table></div>
<figure><figcaption>그림 3. 역별 심야 수요와 이용자당 공급 부족({n_st}개 역)</figcaption>{FIGS}
<div class="legend"><span><i style="background:#111;border-radius:50%"></i>심층 분석 2곳</span><span><i style="background:#555;border-radius:50%"></i>상위 8곳</span><span><i style="background:#a8a8a8;border-radius:50%"></i>우선 검토 영역</span><span><i style="border:1.3px solid #8a8a8a;border-radius:50%"></i>그 외 역</span></div></figure>
<div class="tw"><table><tr><th>순위</th><th>역</th><th>24시 이후 하차<br>(중앙값, 명/일)</th><th>00시 이후<br>기록 노선</th><th>01시 이후<br>기록 노선</th><th>올빼미<br>노선</th><th>자정 이후<br>기록 없음</th><th>이용자당<br>공급 부족</th></tr>{q8_rows}</table></div>
<ul>
<li>수요에서 공급을 빼는 방식은 이용객이 많은 역이 상위에 오르고, 수요를 공급으로 나누는 방식은 공급이 0~1개인 소규모 역이 수요가 적어도 상위에 오를 수 있음.<br>두 조건을 동시에 요구하는 4분면 방식으로 치우침을 줄임</li>
<li>사당은 수요({CQ.iloc[0]["D_median"]:,.0f}명)가 크고 공급이 부족한 역이며,<br>강변은 00시 이후 이용 기록이 있는 노선이 {int(CQ.iloc[1]["S1"])}개뿐이고 올빼미 노선이 없는 역임</li>
<li>강변의 수요({CQ.iloc[1]["D_median"]:.1f}명)는 기준({qD:.0f}명)을 근소하게 넘는 경계값이므로, 선정 근거는 공급 부족 순위와 유형 대비에 둠</li>
<li>나머지 상위 6곳(창동·대림·영등포구청·신대방·성수·건대입구)에는 같은 코드로 확장 적용할 수 있음</li>
</ul>
<h3>6-4. 전역 대안 분류와 두 역 심층 판정의 연결</h3>
<div class="tw"><table><tr><th>전역 분류(시 단위)</th><th>심층 판정: 연결</th><th>15분 이내 부족</th><th>15분 초과 부족</th><th>계</th></tr>{br_rows}</table></div>
<ul>
<li>두 역에서 대안① 후보로 분류된 {a1_n}개 노선-방향 중, 실제로 연결되거나 15분 이내 조정으로 해결되는 것은 {a1_ok}개임</li>
<li>전역 분류는 방향을 구분하지 않으므로, '01시 이후 이용 기록'으로 분류된 노선도 반대 방향은 일찍 끝날 수 있음({int(BR.loc["01시 이후 이용 기록"].sum())}개 중 {int(BR.loc["01시 이후 이용 기록", "15분 초과"])}개가 15분 초과 부족)</li>
<li>00시대 이용 기록 노선도 지하철 최종 막차(00:54~01:02)와는 30분 넘게 어긋나는 경우가 많아,<br>두 역에서는 대안①만으로 부족하고 대안②(연계 막차 1회 추가)가 중심이 됨</li>
</ul>
<h3>6-5. 지하철 평일 막차 도착(2025-09-30 시각표)</h3>
<div class="two">
<div class="tw"><table><tr><th>역</th><th>호선</th><th>방향</th><th>막차 도착</th></tr>{rows_sub}</table></div>
<ul>
<li>자정 이후 도착 열차: 사당 {len(after0["사당"])}편, 강변 {len(after0["강변"])}편</li>
<li>필요 출발 시각(최종 막차 + 6분): 사당 {hhmm(st_last.loc["사당", "last_max"] + TRANSFER)}, 강변 {hhmm(st_last.loc["강변", "last_max"] + TRANSFER)}</li>
<li>토·일요일은 막차가 자정 전후(23:45~00:00)로 1시간 이르므로, 평일만 분석함</li>
</ul></div>
<h3>6-6. 연계 판정</h3>
<figure><figcaption>그림 4. 23시 이후 지하철 도착 열차와 귀가방향 버스 막차(추정)</figcaption>{FIG1}
<div class="legend"><span><i style="background:#111;width:2px;height:12px"></i>자정 이후 도착 열차</span><span><i style="background:#b5b5b5;width:2px;height:12px"></i>23시대 도착 열차</span><span><i style="background:#111;border-radius:50%"></i>연계 막차 추가 대상</span><span><i style="border:1.3px solid #8a8a8a;border-radius:50%"></i>그 외 노선-방향</span><span><i style="border-left:2px dashed #111;width:0;height:12px"></i>필요 출발 시각</span></div></figure>
<div class="tw"><table><tr><th>역</th><th>귀가 노선-방향</th><th>연결</th><th>경계(±10분)</th><th>30분 이내 부족</th><th>30분 초과 부족</th><th>부족 중앙값</th><th>자정 이후 열차</th></tr>{rows_j}</table></div>
<ul>
<li>부족 시간 0 이하로 연결되는 {n_conn}행은 모두 사당이며, 군포 방면 540번과 남태령 기점으로 들어가는 3개 노선임.<br>이 중 {n_conn - n_sure}행은 여유가 10분 이내라 표에서는 '경계'로 분류함</li>
<li>강변은 {int((H["역"] == "강변").sum())}행 전부가 30분 넘게 부족하며, 올빼미버스도 역 300m 안을 지나지 않음</li>
<li>15분 이내 연장으로 해결되는 행은 {n_le15}개뿐임 → '5~15분 조정' 가설은 두 역에서 성립하지 않음</li>
<li>위 표는 정류소를 포함한 {n_H}행 기준임. 같은 노선-방향을 합친 고유 {n_u}개 기준으로는 연결 {n_conn_u}개, 부족 중앙값 {gap_med_u:.0f}분으로 결론이 같음</li>
</ul>
<h3>6-7. 연장 폭과 연결률</h3>
<figure><figcaption>그림 5. 버스가 역 정류소를 떠나는 시각에 따른 자정 이후 열차 연결률</figcaption>{FIG3}
<div class="legend"><span><i style="background:#111;height:3px"></i>사당</span><span><i style="border-top:3px dashed #8a8a8a;height:0"></i>강변</span><span><i style="background:#111;border-radius:50%"></i>5530번(사당)</span><span><i style="border:2px solid #111;border-radius:50%"></i>광진05번(강변)</span></div></figure>
<ul>
<li>5530번은 현재 자정 이후 열차와 연결되지 않으며, +30분이면 {pct(tp5530["연결률 후"], 0)}, 전 열차 연결에는 +{int(tp5530["전 열차 연결 필요 연장"])}분이 필요함</li>
<li>연결률은 00:30~01:00 구간에서 급하게 오르므로, 30분 연장으로는 절반 이하만 연결됨</li>
</ul>
<h3>6-8. 우선 조정안(탐욕 선택)과 조합 전수 비교</h3>
<figure><figcaption>그림 6. 조정 노선-방향 수에 따른 하류 정류소 커버율</figcaption>{FIG4}
<div class="legend"><span><i style="background:#111;height:3px"></i>채택 구간(증가 5%p 이상)</span><span><i style="background:#b5b5b5;height:3px"></i>추가 조정 시 참고</span></div></figure>
<div class="tw"><table><tr><th>역</th><th>순서</th><th>노선(유형)</th><th>역 정류소 → 방면</th><th>현재 통과<br>(추정)</th><th>연계 막차<br>출발</th><th>부족</th><th>+30분 시<br>연결률</th><th>+60분 시<br>연결률</th><th>커버<br>증가</th><th>22~23시<br>승차(명/일)</th><th>우선순위</th></tr>{rows_ch}</table></div>
<ul>
<li>조정 대상 유형: 마을 {type_cnt.get("마을", 0)} · 지선 {type_cnt.get("지선", 0)} · 간선 {type_cnt.get("간선", 0)}</li>
<li>사당 차순위: 502번(+{full_order["사당"][4][1] * 100:.1f}%p), 5530번(+{gain5530 * 100:.1f}%p).<br>5530번은 하류 목적지의 {pct(r5530["행커버"], 0)}가 540번 등으로 이미 연결되어 추가 효과가 작음</li>
<li>강변 차순위: 3212번(+{full_order["강변"][5][1] * 100:.1f}%p)</li>
</ul>
<h3>6-8-1. 조합 전수 비교와 가중치 민감도</h3>
<div class="tw"><table><tr><th>조정 노선 수 k</th><th>사당 탐욕 선택</th><th>사당 전수 최대</th><th>강변 탐욕 선택</th><th>강변 전수 최대</th></tr>{ex_rows}</table></div>
<ul>
<li>후보 노선-방향(사당 {n_cand["사당"]}개, 강변 {n_cand["강변"]}개)에서 k개를 고르는 모든 조합을 비교한 결과,<br>탐욕 선택의 커버율이 k=1~6 전 구간에서 전수 최대와 같음 → 채택안({cov["사당"][2]}개·{cov["강변"][2]}개)은 같은 노선 수에서 가장 높은 커버율임</li>
<li>목표 커버율별 필요 노선 수(전수): 80% → 사당 {need_k(EX["사당"], .8)}개·강변 {need_k(EX["강변"], .8)}개, 85% → 사당 {need_k(EX["사당"], .85)}개·강변 {need_k(EX["강변"], .85)}개, 90% → 사당 {need_k(EX["사당"], .9)}개·강변 {need_k(EX["강변"], .9)}개<br>{"; ".join(f"{s_} {t_ * 100:.0f}% 기준: {near_note(EX[s_], t_)}" for s_ in ST for t_ in (.8, .85, .9) if near_note(EX[s_], t_)) or "표의 반올림값과 목표 판정이 어긋나는 경우 없음"}(표는 소수 첫째 자리 반올림)</li>
<li>중단 기준 5%p는 운영 효율 가정이며, 3%p로 낮추면 사당 {pick_at("사당", .03)}개·강변 {pick_at("강변", .03)}개, 10%p로 높이면 사당 {pick_at("사당", .10)}개·강변 {pick_at("강변", .10)}개가 됨</li>
</ul>
<h3>가중치 민감도: 채택안을 고정했을 때와 다시 골랐을 때</h3>
<div class="tw"><table><tr><th>역</th><th>노선 집합</th><th>노선</th><th>승차 가중 커버율</th><th>하차 가중 커버율</th></tr>{fix_rows}</table></div>
<ul>
<li>하차 가중치 = 하류 정류소별 22~00시 하차 일평균. 같은 노선-방향이 역 주변 두 정류소에서 나타나도 하류 정류소의 하차 기록은 한 번만 넣음(중복 제거 후 수요점 사당 {len(MA["사당"]["pts"])}개, 강변 {len(MA["강변"]["pts"])}개)</li>
<li>채택안({n_ch}개)을 고정한 채 하차 가중으로 다시 계산하면 커버율은 사당 {pct(FIX["사당"]["b_on_a"])}, 강변 {pct(FIX["강변"]["b_on_a"])}임.<br>같은 수의 노선으로 하차 가중에서 얻을 수 있는 최대값(전수, 사당 {pct(FIX["사당"]["a_best_k"])}·강변 {pct(FIX["강변"]["a_best_k"])})보다 {fix_gap["사당"]:.1f}·{fix_gap["강변"]:.1f}%p 낮음 → 채택안은 다른 수요 가정에서도 커버율의 대부분을 유지함</li>
<li>두 가중치에서 모두 선택된 노선은 사당 {"·".join(common_sel["사당"])}, 강변 {"·".join(common_sel["강변"])}({n_common}개)이며, 이를 1순위로 둠.<br>승차 가중에서만 선택된 사당 {"·".join(dep_b["사당"]) or "없음"}, 강변 {"·".join(dep_b["강변"]) or "없음"}({n_ch - n_common}개)은 2순위이며, 하차 가중에서는 대신 사당 {"·".join(dep_a["사당"]) or "없음"}, 강변 {"·".join(dep_a["강변"]) or "없음"}이 들어옴</li>
<li>하차 가중치도 그 정류소에서 내리는 모든 승객을 세므로 목적지 수요의 대리지표에 그침.<br>역에서 출발하는 승객의 목적지 분포(교통카드 환승 연계 자료)가 있어야 2순위 노선을 확정할 수 있음</li>
</ul>
<figure><figcaption>그림 7. 하류 정류소 커버 지도(역 중심, 점은 귀가방향 버스의 하류 정류소)</figcaption>{FIG5}
<div class="legend"><span><i style="background:#111;border-radius:50%"></i>조정으로 새로 연결</span><span><i style="background:#b5b5b5;border-radius:50%"></i>이미 연결(기존 노선·올빼미)</span><span><i style="border:1px solid #9a9a9a;border-radius:50%"></i>미연결</span><span><i style="background:#555;height:2px"></i>1순위 노선(두 가중치 공통)</span><span><i style="border-top:2px dashed #555;height:0"></i>2순위 노선(승차 가중 의존)</span></div></figure>
<details><summary>귀가방향 {n_H}행(고유 노선-방향 {n_u}개) 전체 판정표</summary>
<div class="tw"><table><tr><th>역</th><th>노선</th><th>유형</th><th>정류소</th><th>방향</th><th>방면</th><th>기점 막차</th><th>추정</th><th>API(검증)</th><th>부족(분)</th><th>연결률</th><th>22~23시 승차/일</th><th>00시 승차/월</th><th>판정</th><th>조치</th></tr>{rows_all}</table></div></details>
<h3>6-9. 팀 제안안 검증</h3>
<div class="tw"><table><tr><th>역</th><th>노선</th><th>방면</th><th>현재 통과(추정)</th><th>팀 제안</th><th>개선 후</th><th>자정 이후 열차 연결률</th><th>전 열차 연결 필요</th><th>API 예정(참고)</th></tr>{rows_tp}</table></div>
<ul>
<li>팀 제안의 방향(자정 전 통과로 추정된 노선의 막차 연장)은 타당하며, 강변 3개 노선은 본 분석 상위 3개와 일치함</li>
<li>다만 +30분은 1단계 조치로 보고, 최종 목표는 필요 출발 시각에 맞춘 연계 막차 1회 추가로 설정함</li>
</ul>
<h3>6-10. 민감도</h3>
<div class="tw"><table><tr><th>조건</th><th>결과</th></tr>{rows_sens}</table></div>
<ul>
<li>환승 시간, 속도계수, 커버 반경을 바꿔도 조정 노선 수(사당 3~4, 강변 5~6)와 결론이 유지됨</li>
</ul>

<h3>6-11. 대기시간과 수송능력 점검</h3>
<div class="tw"><table><tr><th>역</th><th>자정 이후 열차</th><th>연계 막차 1회(필요 출발 시각)</th><th>평균 대기</th><th>최대 대기</th><th>30분 이내</th><th>연계 막차 2회(최대 대기 최소화)</th><th>최대 대기</th><th>평균 대기</th></tr>
{wait_rows}</table></div>
<ul>
<li>'전 열차 연결'은 시간상 탑승이 가능하다는 뜻이며, 1회만 두면 00시 직후 도착 승객은 1시간 가까이 기다림</li>
<li>연계 막차를 2회로 나누면(사당 {hhmm(WAIT["사당"]["two_t1"])}·{hhmm(WAIT["사당"]["need"])}, 강변 {hhmm(WAIT["강변"]["two_t1"])}·{hhmm(WAIT["강변"]["need"])} 역 정류소 출발) 모든 열차의 대기가 30분 이내가 됨.<br>대기 허용 기준(예: 30분)은 정책 결정 사항이므로 1회·2회 안을 함께 제시함</li>
</ul>
<div class="tw"><table><tr><th>역</th><th>노선</th><th>배분 수요<br>방향 미구분<br>(명/평일)</th><th>배분 수요<br>방향별<br>(명/평일)</th><th>22~23시<br>승차<br>(명/일)</th><th>추가 운행<br>(분)</th><th>2회 시<br>차량</th><th>우선<br>순위</th></tr>{cap_rows}</table></div>
<p class="note">배분 수요 = 역의 평일 24시 이후 하차 중앙값 × 23시 승차 비중(방향 미구분은 역-노선, 방향별은 노선-방향 기준). 추가 운행 = 역에서 방향 끝까지 주행시간 추정(분). 2회 시 차량 = 2회 운영 시 노선당 필요 차량 수</p>
<ul>
<li>배분 수요 시나리오는 상한이 아니라 배분 가정에 따른 추정값이며, 실제 수요는 이보다 클 수도 작을 수도 있음.<br>방향 미구분 값(역-노선)은 반대 방향 승차까지 포함하므로, 정원 검토에는 방향별 값을 씀</li>
<li>차량 정원은 차종(대형·중형)에 따라 달라 운수사 확인이 필요하며, 방향별 시나리오 값이 정원을 넘는 노선은 2회 운행 또는 대형 차량 배차를 검토해야 함</li>
<li>차량 배치: 1회 추가도 기존 막차 차량이 역 정류소에서 부족 시간만큼 대기하거나 차고지에서 별도 차량을 투입해야 하며, 기사 근무시간 연장이 전제됨.<br>2회 운영은 출발 간격(사당 {interval["사당"]:.0f}분·강변 {interval["강변"]:.0f}분)이 편도 운행시간보다 짧아 노선당 차량 2대·기사 2명이 필요함. 왕복이 간격 안에 드는 노선은 {"·".join(veh1_routes) or "없음"}뿐임</li>
<li>추가 운행시간은 4-2 모델로 추정한 역에서 방향 끝까지의 주행시간이며, 회송·대기 시간은 포함하지 않음. 운행비는 운수사 원가 자료가 있어야 산정할 수 있음</li>
</ul>

<h2><span class="no">7</span>정책제안 및 기대효과 <span class="form">양식 항목</span></h2>
<div class="tw"><table><tr><th>대안</th><th>서울 전체 적용 규모</th><th>사당·강변 적용 결과</th><th>근거 · 선례</th></tr>
<tr><td class="b">① 막차 5~15분 조정</td><td>{int(A[0]["노선"])}개 노선 후보<br>(분 단위 확인 후 확정)</td><td>대안① 후보 {a1_n}개 중 15분 이내로 해결되는 것은 {a1_ok}개.<br>팀 제안 +30분은 1단계 조치(자정 이후 열차의 {pct(TP["연결률 후"].min(), 0)}~{pct(TP["연결률 후"].max(), 0)} 연결)</td><td>차량·회차 추가 없이 시행 가능</td></tr>
<tr><td class="b">② 연계 막차 1회 추가</td><td>기준 시나리오 {int(A[1]["노선"])}개 노선<br>({A[1]["link"]:,.0f}명/평일)<br>기준선에 따라 {BS[0][1]}~{BS[3][1]}개</td><td><b>{n_ch}개 노선-방향 = 1순위 {n_common} + 2순위 {n_dep}</b><br>1순위(두 가중치 공통): 사당 {ch1["사당"]}, 강변 {ch1["강변"]}<br>2순위(승차 가중 의존): 사당 {ch2["사당"]}, 강변 {ch2["강변"]}<br>출발 사당 {hhmm(st_last.loc["사당", "last_max"] + TRANSFER)}, 강변 {hhmm(st_last.loc["강변", "last_max"] + TRANSFER)}<br>커버율(승차 가중) 사당 {pct(cov["사당"][1])}, 강변 {pct(cov["강변"][1])}; 채택안 고정 시 하차 가중 사당 {pct(FIX["사당"]["b_on_a"])}, 강변 {pct(FIX["강변"]["b_on_a"])}</td><td>2025년 연말 시내버스 87개 노선 01시 연장 운행 선례</td></tr>
<tr><td class="b">② 운영 형태</td><td>-</td><td>1회(최종 막차 연계)면 최대 대기 사당 {WAIT["사당"]["one_max"]:.0f}분·강변 {WAIT["강변"]["one_max"]:.0f}분,<br>2회({hhmm(WAIT["사당"]["two_t1"])}·{hhmm(WAIT["강변"]["two_t1"])} 추가)면 최대 {WAIT["사당"]["two_max"]:.0f}분·{WAIT["강변"]["two_max"]:.0f}분</td><td>허용 대기시간과 차량 정원에 따라 선택(6-11).<br>2회 운영은 노선당 차량 2대·기사 2명이 전제({"·".join(veh1_routes) or "없음"} 제외)</td></tr>
<tr><td class="b">③ 대체수단 연계</td><td>{int(A[2]["노선"])}개 노선<br>({A[2]["link"]:,.0f}명/평일)</td><td>남은 목적지 사당 {pct(1 - cov["사당"][1])}, 강변 {pct(1 - cov["강변"][1])}.<br>강변은 역 300m 안 올빼미 노선이 없어 심야 DRT 검토 대상</td><td>서울연구원 심야버스 연계형 DRT 제안(2024),<br>국토교통부 DRT 가이드라인(2025-12)</td></tr>
<tr><td class="b">제도</td><td>마을버스 인센티브 평가에 '지하철 막차 연계' 지표 반영<br>(대안② {int(A[1]["노선"])}개 중 마을버스 {int(A[1]["마을"])}개)</td><td>조정 대상 {n_ch}개 중 마을버스 {type_cnt.get("마을", 0)}개</td><td>시–마을버스조합 합의문 4항(서비스 평가 차등 지급), 7항(3개월 후 운행실적 검토·조정)</td></tr>
</table></div>
<h3>기대효과</h3>
<ul>
<li>추가 운행 하루 {n_ch}회(2회 운영 시 {n_ch * 2}회, 노선당 차량 2대 또는 기존 막차 차량의 역 대기가 조건)로 하류 정류소 커버율(승차 가중)을 사당 +{(cov["사당"][1] - cov["사당"][0]) * 100:.1f}%p, 강변 +{(cov["강변"][1] - cov["강변"][0]) * 100:.1f}%p 높임</li>
<li>규모 대리지표: 평일 24시 이후 하차 중앙값 × 커버율 증가 = 사당 약 {f0(gain_ppl["사당"])}명, 강변 약 {f0(gain_ppl["강변"])}명.<br>실제 버스 환승 인원이 아니라 규모 비교용임</li>
<li>이동 격차: 조정 대상 {n_ch}개 중 {type_cnt.get("마을", 0)}개가 마을버스로, 역에서 먼 주거지의 마지막 구간을 이음</li>
<li>탄소중립: 심야 택시·승용차 이동의 대중교통 전환 효과가 있으나,<br>전환율·배출계수 출처를 확보하기 전까지 수치는 제시하지 않음</li>
<li>확장성: 같은 코드로 다른 역에도 적용 가능(역 이름만 바꿔 실행)</li>
</ul>

<h2><span class="no">8</span>AI 기술 활용 <span class="form">양식 항목</span></h2>
<div class="tw"><table><tr><th>사용 서비스</th><th>활용 범위</th><th>주요 프롬프트</th></tr>
<tr><td class="b">Claude (Anthropic, Claude Code)</td><td>공공데이터 수집·정제 코드 작성, 방향 구분·최대 커버 선택 코드 작성,<br>팀 산출물 재계산·검증, 시각화, 보고서 초안</td><td>"이용자에 비해 버스, 지하철 막차가 가장 빠르게 끊기는 곳을 근거가 있는 기준을 잡아서 도출"<br>"4-1, 5-3 진행"(방향 구분·조정 노선 선택 공통 코드)<br>"구글문서 및 파일들 참고하여 최종 산출물 도출, 검증 과정·결과·시각화 포함"</td></tr>
<tr><td class="b">팀원 사용 도구</td><td colspan="2">[제출 전 팀 기입] 막차 추정 모델·정책 사례 조사에 쓴 AI 서비스, 활용 범위, 주요 프롬프트</td></tr>
</table></div>
<ul>
<li>AI가 만든 수치는 원자료 재계산과 API 대조로 검증했으며, 검증에서 드러난 오류(이상치 해석, 방향 구분)는 5절에 기록함</li>
<li>개인정보·비공개 자료는 AI 서비스에 입력하지 않음</li>
</ul>

<h2><span class="no">9</span>한계 및 확인 필요 사항</h2>
<div class="tw"><table><tr><th>등급</th><th>항목</th><th>내용 · 대응</th></tr>
<tr><td><span class="tag">critical</span></td><td>버스 막차는 추정값</td><td>2026년 API 예정값과 평균 절대오차 {err_mae:.0f}분, 최대 {abs(worst["err"]):.0f}분. 이 대조는 자료 간 일관성 확인이며 2025년 실제 운행 검증이 아님.<br>조정 대상 {n_ch}개 노선은 운수사 시간표로 확인 필요</td></tr>
<tr><td><span class="tag">critical</span></td><td>환승 도보 6분</td><td>실측 전 가정임 → 역 출구–정류소 보행시간 현장 측정으로 교체</td></tr>
<tr><td><span class="tag">critical</span></td><td>연결률의 의미</td><td>열차 편수 기준이며 승객 수 기준이 아님(시각표에 열차별 하차 인원이 없음)</td></tr>
<tr><td><span class="tag">critical</span></td><td>API 시점</td><td>2026-09-23 예정값으로 2025년 시각표와 다를 수 있음 → 검증에만 사용</td></tr>
<tr><td><span class="tag">critical</span></td><td>커버율의 의미</td><td>'하류 정류소 커버율(승차 가중)'은 목적지 수요를 직접 재지 않음.<br>채택안을 고정해도 하차 가중 커버율은 사당 {pct(FIX["사당"]["b_on_a"])}·강변 {pct(FIX["강변"]["b_on_a"])}로 유지되나, 2순위 {n_dep}개는 가중치에 따라 달라지므로 교통카드 환승 연계 자료로 확정 필요</td></tr>
<tr><td><span class="tag">critical</span></td><td>이용 기록 등급</td><td>서울 전체 분류는 이용 기록 기준이며 운행 여부와 다름. 기점 막차 기반 추정과 {AGREE["B"] * 100:.0f}% 일치하나, 두 추정 간 일관성일 뿐 실제 운행 정확도는 미확인</td></tr>
<tr><td><span class="tag m">minor</span></td><td>기준선 34명</td><td>비교 범위가 다른 가정값. 기준선을 바꾸면 대안② 노선 수가 {BS[0][1]}~{BS[3][1]}개 범위에서 변함</td></tr>
<tr><td><span class="tag m">minor</span></td><td>우선 조정안</td><td>탐욕 선택 결과가 k≤6 전수 비교와 일치하나, 7개 이상 조합은 비교하지 않음</td></tr>
<tr><td><span class="tag m">minor</span></td><td>방향 구분</td><td>502번은 종점명을 찾지 못해 기점에서 가장 먼 정류소를 회차점으로 봄</td></tr>
<tr><td><span class="tag m">minor</span></td><td>운영 비용·정원</td><td>추가 운행시간과 2회 운영 시 필요 차량 수는 추정했으나, 운행비·회송·인력(운수종사자 8시간 휴식 규정)과 차량 정원은 운수사 자료가 필요함.<br>배분 수요 시나리오는 상한이 아닌 추정값임</td></tr>
<tr><td><span class="tag m">minor</span></td><td>기존 수치와의 차이</td><td>정류소 위치를 2025-12-09판으로 통일해 서울 전체 수치를 다시 계산함.<br>팀 문서의 기존 값(대안② 44개·2,974명, 대안③ 223개·2,265명, 218개 역)과 소폭 다름</td></tr>
<tr><td><span class="tag m">minor</span></td><td>대안 규모의 단위</td><td>서울 전체 대안 규모는 시 단위 승하차 기록으로 판정한 것이며, 분 단위 막차와 방향은 두 역에서만 확인함</td></tr>
<tr><td><span class="tag m">minor</span></td><td>코드 저장소</td><td>[제출 전 팀 기입] 공개 저장소 생성 후 링크 기입(재현 순서는 문서 끝 참고)</td></tr>
</table></div>

<h2><span class="no">10</span>분석도구 및 참고문헌 <span class="form">양식 항목</span></h2>
<ul class="src">
<li>분석도구: Python 3(pandas, numpy), 시각화는 SVG 직접 작성</li>
<li>Church, R., &amp; ReVelle, C. (1974). The maximal covering location problem. <i>Papers in Regional Science</i>, 32(1), 101–118. {L("링크", "https://doi.org/10.1111/j.1435-5597.1974.tb00902.x")}</li>
<li>Currie, G. (2010). Quantifying spatial gaps in public transport supply based on social needs. <i>Journal of Transport Geography</i>, 18(1), 31–41. {L("링크", "https://www.sciencedirect.com/science/article/abs/pii/S0966692308001518")}</li>
<li>Jiao, J., &amp; Dillivan, M. (2013). Transit deserts: The gap between demand and supply. <i>Journal of Public Transportation</i>, 16(3), 23–39. {L("링크", "https://digitalcommons.usf.edu/jpt/vol16/iss3/2/")}</li>
<li>Fransen, K., Neutens, T., Farber, S., De Maeyer, P., Deruyter, G., &amp; Witlox, F. (2015). Identifying public transport gaps using time-dependent accessibility levels. <i>Journal of Transport Geography</i>, 48, 176–187. {L("링크", "https://www.sciencedirect.com/science/article/abs/pii/S0966692315001714")}</li>
<li>서울연구원(2024). 서울시 수요응답형 이동 서비스(DRT) 도입 방안. {L("링크", "https://eiec.kdi.re.kr/policy/domesticView.do?ac=0000186076")}</li>
<li>Transportation Research Board. Transit Capacity and Quality of Service Manual, 2nd Ed. (TCRP Report 100), Part 3. {L("링크", "https://onlinepubs.trb.org/onlinepubs/tcrp/docs/tcrp100/Part3.pdf")}</li>
<li>서울특별시–마을버스운송사업조합 추가 공동 합의(2025-12-18). {L("링크", "https://opengov.seoul.go.kr/sanction/35062084")}</li>
<li>뉴데일리, 연말 버스 막차 새벽 1시까지 연장(2025-12-03). {L("링크", "https://www.newdaily.co.kr/site/data/html/2025/12/03/2025120300080.html")}</li>
<li>여객자동차 운수사업법 시행규칙 제44조의6(운수종사자의 휴식시간 보장). {L("링크", "https://www.law.go.kr/LSW//lsLawLinkInfo.do?lsJoLnkSeq=1000573147&lsId=007709&chrClsCd=010202&print=print")}</li>
</ul>
<div class="foot">재현 순서: code/final/f0_citywide_2025.py → code/team/run_4-1_direction_사당강변.py → code/final/f1_final_analysis.py → code/final/f2_build_final_html.py · 상세 표: out/final/최종_사당강변_막차연계_처방표.xlsx</div>
"""
JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('circle[data-tip]').forEach(c=>{
 c.addEventListener('mousemove',e=>{tip.innerHTML=c.dataset.tip.split('|').join('<br>');
  tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';});
 c.addEventListener('mouseleave',()=>tip.style.display='none');});
"""
page = ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>사당·강변 막차 연계 분석보고서</title><style>' + fonts + CSS + '</style></head><body><div class="page">' + BODY +
        '</div><div id="tip"></div><script>' + JS + '</script></body></html>')
open(OUT, "w", encoding="utf-8").write(page)
print("저장:", os.path.abspath(OUT), f"{os.path.getsize(OUT) / 1e6:.1f}MB")
print(f"고유 {n_conn_u}/{n_u} 중앙값 {gap_med_u:.1f} | 공통선택 {common_sel} | 대기 {[(s_, round(WAIT[s_]['one_max']), round(WAIT[s_]['two_max'])) for s_ in ST]}")
print(f"연결 {n_conn}/{n_H} (확실 {n_sure}, 경계 {n_edge}) | 중앙값 {gap_med:.1f} | 15분이내 {n_le15} | 선택 {n_ch} {type_cnt} | 커버 {cov}")
print(f"검증 MAE {err_mae:.1f} in10 {in10:.3f} in20 {in20:.3f} worst {worst['노선']} {worst['err']:.0f} | 이상치 {len(outl)} big {len(outl_big)} api후 {outl_api_after} | 재현 {repro_ok}/{repro_n}, H {repro_H}/{len(Hrep)}")
print(f"5530: rank {rank5530}, gain {gain5530:.3f}, 행커버 {r5530['행커버']:.2f} | 효과 {gain_ppl}")
