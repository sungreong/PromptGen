# Prompt Generator 프로젝트 가이드

## 프로젝트 개요
이 프로젝트는 Streamlit을 활용하여 사용자 맞춤형 프롬프트 생성기를 만드는 것입니다. 사용자는 프롬프트를 작성하고, 특정 형식으로 입력하여 결과를 얻을 수 있습니다. 이 애플리케이션은 사용자 인증, JSON 스키마 생성 및 다운로드 기능을 포함합니다.

## 프로젝트 목적
코드를 잘 모르는 사람이 프롬프트를 활용하여 창의적인 아이디어를 생성하고, 다양한 작업을 자동화할 수 있도록 돕는 것입니다. 이 애플리케이션은 사용자 친화적인 인터페이스를 제공하여 누구나 쉽게 사용할 수 있도록 설계되었습니다.

## 기능
- 사용자 로그인 정보 입력
- 프롬프트 폴더 단위 관리
- 프롬프트 작성 및 입력
- JSON 스키마 생성
- JSON 파일 다운로드
- 폴더 내 파일 목록 표시 및 다운로드

## 설치 방법
1. **필수 패키지 설치**
    ```bash
    poetry add $(cat requirements.txt)
    ```
2. **API 키 설정**
   OpenAI API 키를 입력하여 환경 변수를 설정합니다.

## 사용 방법
1. 애플리케이션을 실행합니다.
   ```bash
   streamlit run prompt_flow.py
   ```

   ```bash
   streamlit run prompt_example_main.py
   ```
2. 로그인 정보를 입력합니다.
3. 프롬프트를 작성하고 `{}`를 사용하여 입력을 추가합니다.
4. JSON 스키마를 생성하고 다운로드합니다.
5. 폴더 내의 파일 목록을 확인하고 필요한 파일을 다운로드합니다.

## 코드 구조
- `prompt_example_main.py`: 프롬프트 생성 및 파일 다운로드 기능을 포함합니다.
- `prompt_flow.py`: API 키 입력 및 JSON 스키마 생성 기능을 포함합니다.

## 기여 방법
기여를 원하시는 분은 다음 단계를 따라 주세요:
1. Fork 이 저장소
2. 새로운 브랜치를 생성합니다.
   ```bash
   git checkout -b feature/YourFeature
   ```
3. 변경 사항을 커밋합니다.
   ```bash
   git commit -m "Add your feature"
   ```
4. 브랜치를 푸시합니다.
   ```bash
   git push origin feature/YourFeature
   ```
5. Pull Request를 생성합니다.

# 화면

![](./static/image.png)

## 라이센스
이 프로젝트는 MIT 라이센스 하에 배포됩니다.

