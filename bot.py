import os
import io
import aiohttp
import base64
import logging
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from datetime import datetime

# AI SDKs
import openai
import anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 설정 및 초기화 ---
load_dotenv()

# 로깅 설정 (P2: 날짜별 로그는 아니지만 기본 로깅 구현)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 인텐트 설정
intents = discord.Intents.default()
intents.message_content = True # 메시지 내용 읽기 권한 필수

bot = commands.Bot(command_prefix="!", intents=intents)

# 시스템 프롬프트 (P2: 냥체 페르소나)
SYSTEM_PROMPT = """
당신은 디스코드 AI 봇입니다. 사용자의 질문에 친절하고 정확하게 답변하세요.
가능하다면 문장 끝에 '냥', '이다냥' 등을 붙여서 귀여운 고양이 말투(냥체)를 사용해주세요.
"""

# 세션 관리 (채널 ID를 키로 사용)
chat_sessions = {}

# 최대 컨텍스트 토큰 (근사치: 문자 수 기준 50,000 * 4 = 200,000자 정도로 제한)
MAX_CONTEXT_CHARS = 200000 

class ChatSession:
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name
        self.history = [] # 대화 기록
        self.is_processing = False # P1: 응답 중인지 확인
        
        # 시스템 메시지 초기화
        if provider == "openai":
            self.history.append({"role": "system", "content": SYSTEM_PROMPT})
        # Claude와 Gemini는 시스템 프롬프트를 별도 파라미터로 처리하거나 첫 메시지에 포함

    def add_user_message(self, content, image_data=None):
        msg = {"role": "user", "content": content}
        if image_data:
            msg["image_data"] = image_data # (mime_type, base64_data 또는 bytes)
        self.history.append(msg)
        self.trim_history()

    def add_assistant_message(self, content):
        self.history.append({"role": "assistant", "content": content})
        self.trim_history()

    def trim_history(self):
        # P1: 컨텍스트 관리 (단순 문자 길이로 근사치 계산)
        total_chars = sum(len(str(m["content"])) for m in self.history)
        while total_chars > MAX_CONTEXT_CHARS and len(self.history) > 1:
            # 시스템 프롬프트(인덱스 0)는 유지하려고 노력 (OpenAI 경우)
            idx_to_remove = 1 if self.history[0].get("role") == "system" else 0
            if idx_to_remove < len(self.history):
                removed = self.history.pop(idx_to_remove)
                total_chars -= len(str(removed["content"]))

# --- AI API 핸들러 ---

async def generate_response(session: ChatSession):
    try:
        if session.provider == "openai":
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            messages = []
            for msg in session.history:
                content = msg["content"]
                # 이미지 처리 (OpenAI Vision)
                if msg.get("image_data"):
                    mime, data = msg["image_data"]
                    # OpenAI는 base64 문자열 필요
                    b64_str = base64.b64encode(data).decode('utf-8')
                    content = [
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_str}"}}
                    ]
                messages.append({"role": msg["role"], "content": content})
            
            response = await client.chat.completions.create(
                model=session.model_name,
                messages=messages,
                service_tier="flex"
            )
            return response.choices[0].message.content

        elif session.provider == "claude":
            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            messages = []
            for msg in session.history:
                if msg["role"] == "system": continue # Claude 시스템 프롬프트는 별도
                
                content = msg["content"]
                if msg.get("image_data"):
                    mime, data = msg["image_data"]
                    b64_str = base64.b64encode(data).decode('utf-8')
                    content = [
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64_str}},
                        {"type": "text", "text": msg["content"]}
                    ]
                messages.append({"role": msg["role"], "content": content})

            response = await client.messages.create(
                model=session.model_name,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages
            )
            return response.content[0].text

        elif session.provider == "google":
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(
                session.model_name,
                system_instruction=SYSTEM_PROMPT
            )
            
            chat_history = []
            last_user_parts = []
            
            # Gemini ChatSession 구조로 변환 (이미지 포함 시 복잡하므로 1회성 generate_content 사용 권장되나 context 유지를 위해 chat 변환)
            # 단순화를 위해 여기서는 전체 히스토리를 매번 generate_content로 보냄
            prompt_parts = []
            for msg in session.history:
                role = "user" if msg["role"] == "user" else "model"
                text = msg["content"]
                if msg.get("image_data"):
                    mime, data = msg["image_data"]
                    # Gemini는 PIL 이미지나 데이터 객체 선호
                    import PIL.Image
                    img = PIL.Image.open(io.BytesIO(data))
                    prompt_parts.append(text)
                    prompt_parts.append(img)
                else:
                    prompt_parts.append(f"{role}: {text}")
            
            # 마지막 프롬프트가 문자열이어야 함
            response = await model.generate_content_async(prompt_parts)
            return response.text

    except Exception as e:
        logger.error(f"API Error: {e}")
        return f"오류가 발생했다냥: {str(e)}"

# --- Discord UI (Model Selection) ---

class ModelSelect(discord.ui.Select):
    def __init__(self, provider):
        options = []
        if provider == "openai":
            # P0: 목록 불러오기 (OpenAI는 모델이 너무 많아 주요 모델만 하드코딩 권장하지만, 로직상 fetch 예시)
            # 여기서는 안정성을 위해 주요 모델만 나열합니다.
            models = ["gpt-5.2", "gpt-5-mini", "gpt-5-nano"]
            for m in models:
                options.append(discord.SelectOption(label=m, value=m))
        elif provider == "claude":
            models = ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5"]
            for m in models:
                options.append(discord.SelectOption(label=m, value=m))
        elif provider == "google":
            models = ["gemini-3-pro-preview", "gemini-3-flash-preview"]
            for m in models:
                options.append(discord.SelectOption(label=m, value=m))

        super().__init__(placeholder="모델을 선택해주세요...", min_values=1, max_values=1, options=options)
        self.provider = provider

    async def callback(self, interaction: discord.Interaction):
        model_name = self.values[0]
        # 세션 시작
        chat_sessions[interaction.channel_id] = ChatSession(self.provider, model_name)
        await interaction.response.send_message(
            f"**{self.provider}**의 **{model_name}** 모델로 채팅을 시작한다냥! `/stop`으로 종료할 수 있다냥.",
            ephemeral=False
        )

class ProviderSelect(discord.ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label="OpenAI", value="openai", description="GPT-5.2 등"),
            discord.SelectOption(label="Anthropic", value="claude", description="Claude Sonnet 4.5 등"),
            discord.SelectOption(label="Google", value="google", description="Gemini 3 Pro 등"),
        ]
        super().__init__(placeholder="AI 제공사를 선택해주세요...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        provider = self.values[0]
        view = discord.ui.View()
        view.add_item(ModelSelect(provider))
        await interaction.response.send_message(f"**{provider}**를 선택했군요. 모델을 골라주세요!", view=view, ephemeral=True)

class StartView(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.add_item(ProviderSelect())

# --- 봇 이벤트 및 명령어 ---

@bot.event
async def on_ready():
    logger.info(f'Logged in as {bot.user} (ID: {bot.user.id})')
    try:
        synced = await bot.tree.sync()
        logger.info(f'Synced {len(synced)} commands')
    except Exception as e:
        logger.error(f'Failed to sync commands: {e}')

@bot.tree.command(name="start", description="AI와의 채팅을 시작합니다.")
async def start_command(interaction: discord.Interaction):
    # P0: 채널당 하나의 세션만 허용 (덮어쓰기 가능 여부 묻지 않고 바로 새 선택창)
    await interaction.response.send_message("어떤 AI와 대화할까요?", view=StartView(), ephemeral=True)

@bot.tree.command(name="stop", description="AI와의 대화를 종료하고 기억을 지웁니다.")
async def stop_command(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    if channel_id in chat_sessions:
        del chat_sessions[channel_id]
        # P1: 컨텍스트 초기화 알림
        await interaction.response.send_message("대화를 종료하고 메모리를 비웠다냥! 다음에 또 보자냥.", ephemeral=False)
    else:
        await interaction.response.send_message("현재 진행 중인 대화가 없다냥.", ephemeral=True)

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    channel_id = message.channel.id
    
    # 세션이 있는 채널인지 확인
    if channel_id not in chat_sessions:
        return

    session = chat_sessions[channel_id]

    # P1: 응답 도중에 온 채팅 무시 (Locking)
    if session.is_processing:
        return

    session.is_processing = True

    try:
        # P1: 채팅 입력중 표시
        async with message.channel.typing():
            # 이미지 처리 (P1)
            image_data = None
            if message.attachments:
                for attachment in message.attachments:
                    if attachment.content_type and attachment.content_type.startswith('image'):
                        image_bytes = await attachment.read()
                        image_data = (attachment.content_type, image_bytes)
                        break # 첫 번째 이미지만 처리
            
            # 사용자 메시지 기록
            user_text = message.content or "(이미지 전송됨)"
            session.add_user_message(user_text, image_data)

            # AI 응답 생성
            response_text = await generate_response(session)
            
            # AI 메시지 기록
            session.add_assistant_message(response_text)

            # P0: 2000자 초과 처리 (파일로 전송)
            if len(response_text) > 2000:
                file_data = io.BytesIO(response_text.encode('utf-8'))
                discord_file = discord.File(file_data, filename=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                await message.channel.send("답변이 너무 길어서 파일로 보낸다냥!", file=discord_file)
            else:
                await message.channel.send(response_text)

    except Exception as e:
        logger.error(f"Error in on_message: {e}")
        await message.channel.send("오류가 발생했다냥. 잠시 후 다시 시도해달라냥.")
    
    finally:
        session.is_processing = False

# 봇 실행
if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_TOKEN")
    if not TOKEN:
        print("Error: .env 파일에 DISCORD_TOKEN이 없습니다.")
    else:
        bot.run(TOKEN)