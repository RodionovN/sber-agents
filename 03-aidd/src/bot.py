import os
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from openai import AsyncOpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен в .env")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY не установлен в .env")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Системный промпт для роли финансового советника
SYSTEM_PROMPT = """Ты финансовый советник. Твоя задача - помогать пользователям с финансовыми вопросами, давать советы по управлению финансами, инвестициям, планированию бюджета и другим финансовым темам. Отвечай профессионально, понятно и дружелюбно."""

# Настройка клиента OpenAI для OpenRouter
openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "AIDD Bot",
    },
)

async def get_llm_response(user_message: str) -> str:
    """Отправляет запрос к LLM через OpenRouter и возвращает ответ"""
    try:
        response = await openai_client.chat.completions.create(
            model="tngtech/deepseek-r1t2-chimera:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка при запросе к LLM: {e}", exc_info=True)
        error_str = str(e)
        
        # Обработка специфичных ошибок
        if "402" in error_str or "Insufficient credits" in error_str:
            return "Извините, на аккаунте OpenRouter недостаточно кредитов. Пожалуйста, пополните баланс на https://openrouter.ai/settings/credits"
        elif "401" in error_str or "Unauthorized" in error_str:
            return "Ошибка авторизации. Проверьте правильность API ключа OpenRouter."
        elif "429" in error_str or "rate limit" in error_str.lower():
            return "Превышен лимит запросов. Попробуйте позже."
        else:
            return f"Извините, произошла ошибка при обработке запроса. Попробуйте позже."

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Привет! Я финансовый советник. Задай мне вопрос о финансах, инвестициях, планировании бюджета или других финансовых темах, и я помогу тебе.")

@dp.message()
async def message_handler(message: Message):
    if not message.text:
        await message.answer("Пожалуйста, отправьте текстовое сообщение.")
        return
    
    await message.answer("Обрабатываю запрос...")
    response = await get_llm_response(message.text)
    await message.answer(response)

async def main():
    logger.info("Бот запущен")
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

