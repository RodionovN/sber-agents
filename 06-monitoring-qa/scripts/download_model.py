#!/usr/bin/env python3
"""
Скрипт для загрузки моделей HuggingFace с обработкой ошибок сети
"""
import logging
import time
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model(repo_id: str, revision: str = "main", max_retries: int = 5):
    """
    Загрузка модели HuggingFace с retry логикой
    
    Args:
        repo_id: ID репозитория модели (например, "aroxima/multilingual-e5-large-instruct")
        revision: Ревизия модели (default: "main" или "latest")
        max_retries: Максимальное количество попыток
    """
    try:
        from huggingface_hub import snapshot_download, login
        from huggingface_hub.utils import HfHubHTTPError
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        logger.error("huggingface_hub не установлен. Установите: pip install huggingface_hub")
        sys.exit(1)
    
    # Устанавливаем увеличенные таймауты для загрузки
    import os
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10 минут таймаут
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Отключаем предупреждения о symlinks
    
    # Пробуем загрузить с retry
    retry_delay = 10  # Начальная задержка в секундах
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Попытка {attempt + 1}/{max_retries} загрузки модели {repo_id}...")
            logger.info("Это может занять несколько минут, особенно для больших моделей...")
            
            # Загружаем только необходимые файлы для embeddings (исключаем большие ONNX/OpenVINO модели)
            # Для embeddings нужны: config.json, tokenizer файлы, model.safetensors или pytorch_model.bin
            local_dir = snapshot_download(
                repo_id=repo_id,
                revision=revision if revision != "latest" else "main",
                local_dir=None,  # Используем кеш по умолчанию
                resume_download=True,  # Продолжаем загрузку при прерывании
                local_dir_use_symlinks=False,
                ignore_patterns=["*.onnx", "openvino/*"]  # Исключаем большие ONNX и OpenVINO файлы
            )
            
            logger.info(f"✅ Модель успешно загружена: {repo_id}")
            logger.info(f"Расположение: {local_dir}")
            return local_dir
            
        except RepositoryNotFoundError as e:
            error_msg = str(e)
            logger.error(
                f"❌ Модель не найдена или требуется аутентификация.\n"
                f"Репозиторий: {repo_id}\n"
                f"Ошибка: {error_msg}\n\n"
                f"Возможные причины:\n"
                f"1. Модель не существует под таким именем\n"
                f"2. Модель является приватной и требует токен доступа\n"
                f"3. Неправильное имя модели\n\n"
                f"Рекомендации:\n"
                f"1. Проверьте правильность имени модели на https://huggingface.co/models\n"
                f"2. Если это приватная модель, получите токен на https://huggingface.co/settings/tokens\n"
                f"3. Авторизуйтесь: huggingface-cli login\n"
                f"4. Попробуйте альтернативные модели:\n"
                f"   - intfloat/multilingual-e5-large\n"
                f"   - intfloat/multilingual-e5-base\n"
                f"   - intfloat/multilingual-e5-small\n"
                f"   - intfloat/multilingual-e5-large-instruct"
            )
            sys.exit(1)
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Проверяем тип ошибки
            if ("ConnectionError" in error_msg or 
                "Read timed out" in error_msg or 
                "timeout" in error_msg.lower() or
                "HTTPSConnectionPool" in error_msg):
                
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Ошибка сети при загрузке (попытка {attempt + 1}/{max_retries}). "
                        f"Повтор через {retry_delay} секунд..."
                    )
                    logger.debug(f"Детали ошибки: {error_msg[:200]}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Экспоненциальная задержка
                    continue
                else:
                    logger.error(
                        f"❌ Не удалось загрузить модель после {max_retries} попыток.\n"
                        f"Ошибка: {error_msg}\n\n"
                        f"Рекомендации:\n"
                        f"1. Проверьте интернет-соединение\n"
                        f"2. Попробуйте позже (возможны проблемы с HuggingFace Hub)\n"
                        f"3. Используйте VPN если HuggingFace недоступен\n"
                        f"4. Попробуйте загрузить модель вручную через браузер"
                    )
                    sys.exit(1)
            elif ("401" in error_msg or 
                  error_type == "RepositoryNotFoundError" or
                  "Unauthorized" in error_msg or
                  "Repository Not Found" in error_msg):
                logger.error(
                    f"❌ Модель не найдена или требуется аутентификация.\n"
                    f"Репозиторий: {repo_id}\n"
                    f"Ошибка: {error_msg}\n\n"
                    f"Возможные причины:\n"
                    f"1. Модель не существует под таким именем\n"
                    f"2. Модель является приватной и требует токен доступа\n"
                    f"3. Неправильное имя модели\n\n"
                    f"Рекомендации:\n"
                    f"1. Проверьте правильность имени модели на https://huggingface.co/models\n"
                    f"2. Если это приватная модель, получите токен на https://huggingface.co/settings/tokens\n"
                    f"3. Авторизуйтесь: huggingface-cli login\n"
                    f"4. Попробуйте альтернативные модели:\n"
                    f"   - intfloat/multilingual-e5-large\n"
                    f"   - intfloat/multilingual-e5-base\n"
                    f"   - intfloat/multilingual-e5-small\n"
                    f"   - intfloat/multilingual-e5-large-instruct"
                )
                sys.exit(1)
            else:
                # Другие ошибки - пробрасываем
                logger.error(f"Неожиданная ошибка ({error_type}): {error_msg}")
                raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python scripts/download_model.py <repo_id> [revision]")
        print("Пример: python scripts/download_model.py aroxima/multilingual-e5-large-instruct latest")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    revision = sys.argv[2] if len(sys.argv) > 2 else "main"
    
    # Обрабатываем "latest" как "main"
    if revision == "latest":
        revision = "main"
    
    download_model(repo_id, revision)

