# 🤖 Discord Multi-LLM Chatbot

OpenAI(GPT), Anthropic(Claude), Google(Gemini)의 최신 모델을 디스코드에서 한 번에 사용할 수 있는 올인원 AI 챗봇입니다. 대화의 문맥을 기억하며, 이미지를 인식하고, 귀여운 고양이 말투(냥체)를 사용합니다.

## ✨ 주요 기능

- **멀티 모델 지원**: OpenAI, Claude, Gemini 중 원하는 AI를 선택하여 대화할 수 있습니다.
- **슬래시 명령어**: `/start`, `/stop` 명령어로 깔끔하게 제어합니다.
- **문맥 기억**: 채팅 히스토리를 기억하여 연속적인 대화가 가능합니다.
- **이미지 인식**: 이미지를 업로드하고 AI에게 질문할 수 있습니다 (멀티모달 지원).
- **긴 답변 처리**: 디스코드 글자 수 제한(2000자)을 넘는 답변은 자동으로 `.txt` 파일로 변환하여 전송합니다.
- **상태 표시**: AI가 답변을 생성하는 동안 `입력 중...` 상태를 표시하며, 중복 입력을 방지합니다.
- **페르소나**: 친절하고 귀여운 고양이 말투를 사용합니다. 🐱

## 🛠️ 사전 준비 (Prerequisites)

이 봇을 실행하기 위해서는 **Python 3.8 이상**이 설치되어 있어야 합니다.

또한 다음 서비스들의 API 키가 필요합니다 (사용할 서비스의 키만 있어도 됩니다).
- [Discord Developer Portal](https://discord.com/developers/applications) (봇 토큰)
- [OpenAI API](https://platform.openai.com/)
- [Anthropic API](https://console.anthropic.com/)
- [Google AI Studio](https://aistudio.google.com/)

## 📥 설치 및 설정 방법

### 1. 프로젝트 설정
폴더를 생성하고 제공된 `bot.py` 파일을 저장합니다.

### 2. 라이브러리 설치
터미널(CMD, PowerShell)을 열고 필요한 파이썬 패키지를 설치합니다.
```bash
pip install discord.py openai anthropic google-generativeai python-dotenv aiohttp
```

### 3. 환경 변수(.env) 설정
프로젝트 폴더에 `.env` 파일을 생성하고 아래 내용을 입력합니다.
**주의:** API 키 값은 따옴표 없이 입력하세요.

```env
# 디스코드 봇 토큰 (필수)
DISCORD_TOKEN=여기에_봇_토큰_입력

# AI 서비스 API 키 (사용할 서비스만 입력, 나머지는 비워도 됨)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy...
```

### 4. 디스코드 개발자 포털 설정 (중요!)
봇이 채팅을 읽기 위해서는 **Message Content Intent** 권한이 필요합니다.
1. [Discord Developer Portal](https://discord.com/developers/applications) 접속 -> 내 애플리케이션 선택.
2. 좌측 메뉴의 **Bot** 클릭.
3. **Privileged Gateway Intents** 항목에서 **Message Content Intent**를 찾아 스위치를 **ON**으로 켭니다.
4. `Save Changes`를 눌러 저장합니다.
5. `OAuth2` -> `URL Generator`에서 `bot` 체크 -> `Administrator` (또는 필요한 권한) 체크 후 생성된 URL로 서버에 봇을 초대합니다.

## 🚀 실행 방법

터미널에서 아래 명령어로 봇을 실행합니다.

```bash
python bot.py
```
성공적으로 실행되면 터미널에 `Logged in as [봇이름]` 및 `Synced commands` 메시지가 뜹니다.

## 💬 사용 가이드

### 1. 대화 시작하기
채팅 채널에서 슬래시 명령어를 입력합니다.
> `/start`

봇이 **"어떤 AI와 대화할까요?"** 라는 메뉴를 띄웁니다.
1. **제공사 선택**: OpenAI, Anthropic, Google 중 하나를 선택합니다.
2. **모델 선택**: 해당 제공사의 세부 모델(예: GPT-4o, Claude 3.5 Sonnet 등)을 선택합니다.

### 2. 대화하기
모델을 선택하면 채팅 세션이 시작됩니다. 이제 평소처럼 채팅을 치면 봇이 대답합니다.
- 봇은 이전 대화 내용을 기억합니다.
- 사진을 함께 업로드하면 사진 내용을 분석해줍니다.
- 답변을 생성하는 도중에는 추가 질문을 무시합니다.

### 3. 대화 종료하기
대화를 끝내고 싶거나 다른 모델로 바꾸고 싶을 때 입력합니다.
> `/stop`

이 명령어를 입력하면 봇은 기억하고 있던 대화 내용을 삭제(초기화)하고 더 이상 응답하지 않습니다.

## ⚠️ 주의사항

- **비용 문제**: 각 AI API(OpenAI, Claude 등)는 사용량에 따라 비용이 청구될 수 있습니다. (Google Gemini는 일부 무료 티어 제공)
- **컨텍스트 제한**: 대화가 매우 길어지면(약 20만 자 이상) 비용 절감 및 오류 방지를 위해 가장 오래된 대화부터 자동으로 잊어버립니다.
- **API 오류**: API 키가 없거나 잔액이 부족한 경우 봇이 에러 메시지를 출력할 수 있습니다.