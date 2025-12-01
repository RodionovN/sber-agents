#!/usr/bin/env python3
"""
Скрипт для загрузки embedding моделей через Ollama
"""
import subprocess
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_ollama_installed():
    """Проверка установки Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info(f"Ollama установлен: {result.stdout.strip()}")
            return True
        return False
    except FileNotFoundError:
        logger.error("Ollama не найден в PATH")
        return False
    except Exception as e:
        logger.error(f"Ошибка при проверке Ollama: {e}")
        return False

def download_model_ollama(model_name: str):
    """
    Загрузка модели через Ollama
    
    Args:
        model_name: Имя модели в формате Ollama (например, "jeffh/intfloat-multilingual-e5-large")
    """
    if not check_ollama_installed():
        logger.error(
            "❌ Ollama не установлен.\n\n"
            "Установите Ollama:\n"
            "1. Windows: https://ollama.com/download/windows\n"
            "2. macOS: https://ollama.com/download/macos\n"
            "3. Linux: curl -fsSL https://ollama.com/install.sh | sh\n\n"
            "После установки перезапустите терминал и попробуйте снова."
        )
        sys.exit(1)
    
    logger.info(f"Загрузка модели {model_name} через Ollama...")
    logger.info("Это может занять несколько минут, особенно для больших моделей...")
    
    try:
        # Запускаем ollama pull
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=1800  # 30 минут таймаут
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Модель успешно загружена: {model_name}")
            logger.info(f"Вывод: {result.stdout}")
            
            # Проверяем, что модель доступна
            check_result = subprocess.run(
                ["ollama", "show", model_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            if check_result.returncode == 0:
                logger.info(f"✅ Модель доступна и готова к использованию")
                logger.info(f"Информация о модели:\n{check_result.stdout}")
            return True
        else:
            logger.error(f"❌ Ошибка при загрузке модели")
            logger.error(f"Вывод: {result.stderr}")
            logger.error(f"Код возврата: {result.returncode}")
            
            if "not found" in result.stderr.lower() or "404" in result.stderr:
                logger.error(
                    f"\nМодель {model_name} не найдена в реестре Ollama.\n\n"
                    f"Доступные альтернативы для embeddings:\n"
                    f"1. jeffh/intfloat-multilingual-e5-large (рекомендуется)\n"
                    f"2. nomic-embed-text (универсальная модель)\n"
                    f"3. all-minilm (легкая модель)\n\n"
                    f"Проверьте доступные модели на: https://ollama.com/library"
                )
            sys.exit(1)
            
    except subprocess.TimeoutExpired:
        logger.error(
            "❌ Таймаут при загрузке модели (30 минут).\n"
            "Возможно, модель очень большая или проблемы с сетью.\n"
            "Попробуйте запустить команду вручную:\n"
            f"  ollama pull {model_name}"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python scripts/download_model_ollama.py <model_name>")
        print("\nПримеры:")
        print("  python scripts/download_model_ollama.py jeffh/intfloat-multilingual-e5-large")
        print("  python scripts/download_model_ollama.py nomic-embed-text")
        print("\nПримечание:")
        print("  Модель intfloat/multilingual-e5-base недоступна напрямую в Ollama.")
        print("  Используйте jeffh/intfloat-multilingual-e5-large как альтернативу.")
        sys.exit(1)
    
    model_name = sys.argv[1]
    download_model_ollama(model_name)

