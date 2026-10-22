# seoul-lastbus-transfer-gap

서울 지하철–버스 막차 환승 단절 진단과 연계 막차 우선 조정안 (사당·강변역 사례)
— AI와 함께하는 교통문제 해결을 위한 데이터 분석 공모전 (한겨레 × 재단법인 숲과나눔) 제출작 분석 코드

## 실행

```bash
pip install pandas numpy openpyxl
python code/h1_keyfree_v2.py
python code/h2_policy_2025.py
python code/h3_gap_index_2025.py
python code/h4_quadrant_2025.py
python code/h5_stage2_prescription.py
python code/h6_min_adjust_mclp.py
python code/final/f0_citywide_2025.py
python code/final/f1_final_analysis.py
python code/final/f2_build_final_html.py
```

저장소 루트에서 실행합니다. 결과는 `out/`, 최종 보고서 HTML은 루트에 생성됩니다.
`h5`의 API 조회는 `api/cache/`에 저장된 응답(2026-09-23 조회)을 그대로 사용하므로 인증키 없이 재현됩니다.

## 데이터 (`data/`에 아래 파일명으로 저장, 저장소에는 포함하지 않음)

| 파일명 | 자료 | 출처 |
| --- | --- | --- |
| 지하철_열차운행시각표_20250930.csv / _20260902.csv | 서울 도시철도 열차운행시각표 | https://www.data.go.kr/data/15098251/fileData.do |
| 지하철_역별일별시간대별승하차_2025.csv | 역별 일별 시간대별 승하차인원 | https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do |
| 지하철_역좌표_1-8호선_20250814.csv | 1~8호선 역사 좌표 | https://www.data.go.kr/data/15099316/fileData.do |
| 지하철_역좌표_9호선2-3단계_20260131.csv | 9호선 2·3단계 역사 좌표 | 서울교통공사 |
| 버스_정류장별시간대별승하차_2025월별/*.csv, _202608.csv | 버스노선별 정류장별 시간대별 승하차 인원 | https://data.seoul.go.kr/dataList/OA-12913/S/1/datasetView.do |
| 버스_노선별경유정류소순서_20251209.xlsx | 버스노선별 경유정류소 목록정보 | https://data.seoul.go.kr/dataList/OA-15261/L/1/datasetView.do |
| 버스_노선기본정보_기점첫막차_20251209.xlsx | 버스노선 기본정보 | https://data.seoul.go.kr/dataList/OA-15262/S/1/datasetView.do |
| 버스_정류소위치_20251209.xlsx / _20260902.xlsx | 버스정류소 위치정보 | https://data.seoul.go.kr/dataList/OA-15067/S/1/datasetView.do |
| 버스_노선ID목록_20251209.xlsx | 버스노선 ID 정보 | 서울 열린데이터광장 |
| (API) 정류소정보조회 서비스 | 예정 막차 대조 | https://www.data.go.kr/data/15000303/openapi.do |
| 서울시 버스노선별 정류장별 구간 운행속도/시간 | 심야 구간 주행시간 | https://data.seoul.go.kr/dataList/OA-21217/S/1/datasetView.do |

`data/팀산출물/`은 팀이 작성한 중간 산출물(재현 대조용)입니다.

## 구성

- `code/h1`~`h6` : 서울 전체 진단 → 역 선정 → 8개 역 귀가방향 판정 → 최소 조정안
- `code/final/f0`~`f2` : 2025년 자료 기준 최종 분석(사당·강변)과 보고서 HTML 생성
- `code/team/` : 방향 구분(`common_direction`)·최대 커버 탐욕 선택(`common_mclp`) 공통 모듈

데이터의 권리는 각 제공기관에 있으며, 코드는 MIT 라이선스를 따릅니다.
