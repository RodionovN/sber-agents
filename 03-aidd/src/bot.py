import os
import logging
import asyncio
from asyncio import TimeoutError
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

# Хранение истории диалога в памяти (user_id -> список сообщений)
user_history = {}

# Максимальное количество сообщений в истории
MAX_HISTORY_MESSAGES = 10

# Таймаут для запросов к LLM (в секундах)
LLM_TIMEOUT = 60

# Настройка клиента OpenAI для OpenRouter
openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "AIDD Bot",
    },
)

async def get_llm_response(user_id: int, user_message: str) -> str:
    """Отправляет запрос к LLM через OpenRouter с учетом истории диалога"""
    try:
        # Получаем историю пользователя или создаем новую
        history = user_history.get(user_id, [])
        
        # Формируем список сообщений: системный промпт + история + текущее сообщение
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        
        # Запрос с таймаутом
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model="tngtech/deepseek-r1t2-chimera:free",
                messages=messages,
            ),
            timeout=LLM_TIMEOUT
        )
        
        response_text = response.choices[0].message.content
        
        # Сохраняем вопрос и ответ в историю
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response_text})
        
        # Ограничиваем размер истории
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
        
        user_history[user_id] = history
        
        return response_text
    except TimeoutError:
        logger.error(f"Таймаут при запросе к LLM для пользователя {user_id}")
        return "Извините, запрос занял слишком много времени. Попробуйте позже."
    except Exception as e:
        logger.error(f"Ошибка при запросе к LLM для пользователя {user_id}: {e}", exc_info=True)
        error_str = str(e)
        
        # Обработка специфичных ошибок
        if "402" in error_str or "Insufficient credits" in error_str:
            return "Извините, на аккаунте OpenRouter недостаточно кредитов. Пожалуйста, пополните баланс на https://openrouter.ai/settings/credits"
        elif "401" in error_str or "Unauthorized" in error_str:
            return "Ошибка авторизации. Проверьте правильность API ключа OpenRouter."
        elif "429" in error_str or "rate limit" in error_str.lower():
            return "Превышен лимит запросов. Попробуйте позже."
        else:
            return "Извините, произошла ошибка при обработке запроса. Попробуйте позже."

@dp.message(Command("start"))
async def cmd_start(message: Message):
    # Очищаем историю при команде /start
    user_history[message.from_user.id] = []
    await message.answer("Привет! Я финансовый советник. Задай мне вопрос о финансах, инвестициях, планировании бюджета или других финансовых темах, и я помогу тебе.")

@dp.message()
async def message_handler(message: Message):
    if not message.text:
        await message.answer("Пожалуйста, отправьте текстовое сообщение.")
        return
    
    try:
        # Отправляем сообщение "Обрабатываю запрос..." и сохраняем его для редактирования
        status_message = await message.answer("Обрабатываю запрос...")
        
        response = await get_llm_response(message.from_user.id, message.text)
        
        # Редактируем сообщение на ответ от LLM
        try:
            await status_message.edit_text(response)
        except Exception as edit_error:
            logger.warning(f"Не удалось отредактировать сообщение для пользователя {message.from_user.id}: {edit_error}")
            # Если редактирование не удалось, отправляем новое сообщение
            await message.answer(response)
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения от пользователя {message.from_user.id}: {e}", exc_info=True)
        try:
            await message.answer("Извините, произошла ошибка при обработке вашего сообщения. Попробуйте позже.")
        except Exception as send_error:
            logger.error(f"Не удалось отправить сообщение об ошибке пользователю {message.from_user.id}: {send_error}")

async def main():
    logger.info("Бот запущен")
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

