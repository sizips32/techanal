# AI-Powered Technical Analysis Dashboard

## 설치 및 실행 방법

1. 필요한 패키지 설치:

   ```bash
   pip install -r requirements.txt
   ```

2. API 키 설정:

   - `.streamlit/secrets.toml` 파일 생성:
     ```toml
     [gemini]
     api_key = "your_actual_api_key_here"
     ```
   - 또는 환경 변수 설정:
     - Linux/macOS: `export GEMINI_API_KEY="your_actual_api_key_here"`
     - Windows: `set GEMINI_API_KEY=your_actual_api_key_here`

3. 애플리케이션 실행:
   ```bash
   streamlit run technical_analysis.py
   ```

## Streamlit Cloud에 배포하기

Streamlit Cloud에 배포할 경우, 다음과 같이 비밀 값을 설정할 수 있습니다:

1. Streamlit Cloud 대시보드에서 앱 설정으로 이동
2. "Secrets" 섹션 클릭
3. 다음 내용 추가:
   ```toml
   [gemini]
   api_key = "your_actual_api_key_here"
   ```
4. "Save" 버튼 클릭

## 주의사항

- API 키를 GitHub 등의 공개 저장소에 절대 업로드하지 마세요.
- `.gitignore` 파일에 `.streamlit/secrets.toml`이 포함되어 있는지 확인하세요.
- 환경 변수를 사용할 경우, 배포 환경에서도 환경 변수를 설정해야 합니다.


supabase 

npx -y @modelcontextprotocol/server-postgres postgresql://postgres.pafshyoizgtecgrmgwmk:[Marcodex@690418]@aws-0-ap-northeast-2.pooler.supabase.com:5432/postgres
