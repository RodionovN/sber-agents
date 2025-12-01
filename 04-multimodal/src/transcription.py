import logging
import os
import requests
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)

async def transcribe_voice_yandex(audio_file_path: str) -> str:
    """Транскрибация голосового сообщения через Yandex SpeechKit API."""
    try:
        url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
        headers = {"Authorization": f"Api-Key {config.YANDEX_SPEECHKIT_API_KEY}"}
        data = {"lang": "ru-RU", "format": "oggopus"}
        
        with open(audio_file_path, "rb") as audio_file:
            files = {"data": audio_file}
            response = requests.post(url, headers=headers, data=data, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get("result", "")
            logger.info(f"Yandex SpeechKit transcription successful: {len(transcript)} chars")
            return transcript
        else:
            logger.error(f"Yandex SpeechKit API error: {response.status_code} - {response.text}")
            raise Exception(f"Yandex SpeechKit API error: {response.status_code}")
    except Exception as e:
        logger.error(f"Error transcribing with Yandex SpeechKit: {e}", exc_info=True)
        raise

async def transcribe_voice_openai(audio_file_path: str) -> str:
    """Транскрибация голосового сообщения через OpenAI Whisper API."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=config.OPENAI_WHISPER_API_KEY)
        
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ru"
            )
        
        result_text = transcript.text
        logger.info(f"OpenAI Whisper transcription successful: {len(result_text)} chars")
        return result_text
    except Exception as e:
        logger.error(f"Error transcribing with OpenAI Whisper: {e}", exc_info=True)
        raise

async def transcribe_voice_local_whisper(audio_file_path: str) -> str:
    """Транскрибация голосового сообщения через локальный faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        
        # Используем модель base для баланса между качеством и скоростью
        # Можно использовать: tiny, base, small, medium, large
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        device = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Loading Whisper model: {model_size}, device: {device}, compute_type: {compute_type}")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        logger.info(f"Transcribing audio file: {audio_file_path}")
        segments, info = model.transcribe(audio_file_path, language="ru")
        
        # Собираем текст из всех сегментов
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text)
        
        result_text = " ".join(transcript_parts).strip()
        logger.info(f"Local Whisper transcription successful: {len(result_text)} chars, language: {info.language}, probability: {info.language_probability:.2f}")
        return result_text
    except ImportError:
        logger.error("faster-whisper not installed. Install it with: pip install faster-whisper")
        raise ValueError("faster-whisper not installed")
    except Exception as e:
        logger.error(f"Error transcribing with local Whisper: {e}", exc_info=True)
        raise

async def transcribe_voice_server_whisper(audio_file_path: str) -> str:
    """Транскрибация голосового сообщения через серверный Whisper API."""
    try:
        url = f"{config.WHISPER_SERVER_URL}/transcribe"
        logger.info(f"Transcribing via server Whisper API: {url}, file: {audio_file_path}")
        
        # Проверяем что файл существует
        if not Path(audio_file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        file_size = Path(audio_file_path).stat().st_size
        logger.info(f"Audio file size: {file_size} bytes")
        
        with open(audio_file_path, "rb") as audio_file:
            files = {"file": (Path(audio_file_path).name, audio_file, "audio/ogg")}
            data = {"language": "ru"}
            response = requests.post(url, files=files, data=data, timeout=120)
        
        logger.info(f"Server Whisper API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get("text", "")
            logger.info(f"Server Whisper transcription successful: {len(transcript)} chars, text: {transcript[:200]}")
            
            if not transcript or not transcript.strip():
                logger.warning("Server Whisper returned empty transcript")
                raise ValueError("Транскрибация вернула пустой текст")
            
            return transcript
        else:
            logger.error(f"Server Whisper API error: {response.status_code} - {response.text}")
            raise Exception(f"Server Whisper API error: {response.status_code} - {response.text}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}", exc_info=True)
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error connecting to Whisper server: {e}", exc_info=True)
        raise Exception(f"Ошибка подключения к серверу транскрибации: {e}")
    except Exception as e:
        logger.error(f"Error transcribing with server Whisper: {e}", exc_info=True)
        raise

async def transcribe_voice(audio_file_path: str) -> str:
    """Единый интерфейс для транскрибации голосовых сообщений."""
    provider = config.TRANSCRIPTION_PROVIDER.lower() if config.TRANSCRIPTION_PROVIDER else "yandex"
    
    logger.info(f"Transcribing voice message using provider: {provider}")
    
    if provider == "yandex":
        if not config.YANDEX_SPEECHKIT_API_KEY:
            raise ValueError("YANDEX_SPEECHKIT_API_KEY not configured")
        return await transcribe_voice_yandex(audio_file_path)
    elif provider == "openai":
        if not config.OPENAI_WHISPER_API_KEY:
            raise ValueError("OPENAI_WHISPER_API_KEY not configured")
        return await transcribe_voice_openai(audio_file_path)
    elif provider == "local_whisper":
        return await transcribe_voice_local_whisper(audio_file_path)
    elif provider == "server_whisper":
        if not config.WHISPER_SERVER_URL:
            raise ValueError("WHISPER_SERVER_URL not configured")
        return await transcribe_voice_server_whisper(audio_file_path)
    else:
        raise ValueError(f"Unknown transcription provider: {provider}. Supported: yandex, openai, local_whisper, server_whisper")

