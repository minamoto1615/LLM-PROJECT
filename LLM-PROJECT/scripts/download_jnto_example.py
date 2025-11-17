"""
간단 예시: JNTO 통계 페이지에서 'Download Data (CSV)' 우클릭 → 링크 주소 복사
그 URL을 아래에 붙여넣고 실행하면 같은 폴더에 CSV로 저장됨.

이 스크립트는 인터넷이 되는 너 컴퓨터에서 돌려야 한다.
"""

import requests

# TODO: 여기 URL을 네가 JNTO 사이트에서 복사해서 붙여넣기
JNTO_CSV_URL = "https://statistics.jnto.go.jp/example.csv"  # <- 이 줄 수정

def main():
    if "example.csv" in JNTO_CSV_URL:
        raise SystemExit("먼저 JNTO_CSV_URL 을 실제 CSV 주소로 바꿔줘.")
    print("다운로드 중:", JNTO_CSV_URL)
    r = requests.get(JNTO_CSV_URL)
    r.raise_for_status()
    out_name = "jnto_raw.csv"
    with open(out_name, "wb") as f:
        f.write(r.content)
    print("저장 완료:", out_name)

if __name__ == "__main__":
    main()
