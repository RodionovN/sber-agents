"""
Скрипт для скачивания Cross-Encoder модели в локальную папку
Поддерживает несколько методов скачивания
"""
import os
import sys
import argparse
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
LOCAL_PATH = r"D:\Documents\Projects\sber-agents\models\mmarco-mMiniLMv2-L12-H384-v1"


def method_huggingface_hub():
    """Метод 1: Использование huggingface_hub.snapshot_download (рекомендуется)"""
    print("=" * 60)
    print("Метод 1: Использование huggingface_hub.snapshot_download")
    print("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        
        print(f"Скачивание модели {MODEL_NAME} в {LOCAL_PATH}...")
        os.makedirs(LOCAL_PATH, exist_ok=True)
        
        # Скачиваем все файлы модели
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=LOCAL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True  # Продолжить скачивание, если было прервано
        )
        
        print(f"✅ Модель успешно скачана в {LOCAL_PATH}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при скачивании через huggingface_hub: {e}")
        import traceback
        traceback.print_exc()
        return False


def method_transformers():
    """Метод 2: Использование transformers.AutoModel"""
    print("=" * 60)
    print("Метод 2: Использование transformers.AutoModel")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        print(f"Скачивание модели {MODEL_NAME} в {LOCAL_PATH}...")
        os.makedirs(LOCAL_PATH, exist_ok=True)
        
        # Скачиваем модель и токенайзер
        print("Загрузка модели...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        print("Загрузка токенайзера...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Сохраняем в локальную папку
        print(f"Сохранение в {LOCAL_PATH}...")
        model.save_pretrained(LOCAL_PATH)
        tokenizer.save_pretrained(LOCAL_PATH)
        
        print(f"✅ Модель успешно скачана в {LOCAL_PATH}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при скачивании через transformers: {e}")
        import traceback
        traceback.print_exc()
        return False


def method_sentence_transformers():
    """Метод 3: Использование sentence_transformers.CrossEncoder"""
    print("=" * 60)
    print("Метод 3: Использование sentence_transformers.CrossEncoder")
    print("=" * 60)
    
    try:
        from sentence_transformers import CrossEncoder
        
        print(f"Скачивание модели {MODEL_NAME} в {LOCAL_PATH}...")
        os.makedirs(LOCAL_PATH, exist_ok=True)
        
        # Загружаем модель из HuggingFace Hub
        print("Загрузка модели из HuggingFace Hub...")
        model = CrossEncoder(MODEL_NAME)
        
        # Сохраняем модель в локальную папку
        print(f"Сохранение модели в {LOCAL_PATH}...")
        model.save(LOCAL_PATH)
        
        print(f"✅ Модель успешно скачана в {LOCAL_PATH}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при скачивании через sentence_transformers: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_downloaded_files():
    """Проверяет наличие скачанных файлов"""
    print("\n" + "=" * 60)
    print("Проверка скачанных файлов")
    print("=" * 60)
    
    pytorch_files = ['pytorch_model.bin', 'model.safetensors']
    config_files = ['config.json', 'tokenizer_config.json', 'vocab.txt']
    found_files = []
    
    for f in pytorch_files + config_files:
        file_path = os.path.join(LOCAL_PATH, f)
        if os.path.exists(file_path):
            found_files.append(f)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  ✓ {f} ({file_size:.2f} MB)")
    
    if any(f in found_files for f in pytorch_files):
        print(f"\n✅ Модель готова к использованию!")
        print(f"Найдены файлы PyTorch модели: {', '.join([f for f in found_files if f in pytorch_files])}")
        return True
    else:
        print("\n⚠️  Файлы модели PyTorch не найдены.")
        return False


def download_model(method: str = "auto"):
    """Скачивает модель используя указанный метод"""
    print(f"\nСкачивание модели: {MODEL_NAME}")
    print(f"Целевая папка: {LOCAL_PATH}")
    print(f"Размер модели: ~471MB\n")
    
    methods = {
        "1": ("huggingface_hub", method_huggingface_hub),
        "2": ("transformers", method_transformers),
        "3": ("sentence_transformers", method_sentence_transformers),
    }
    
    if method == "auto":
        # Пробуем все методы по порядку
        for method_name, method_func in methods.values():
            print(f"\nПробуем метод: {method_name}...")
            if method_func():
                if check_downloaded_files():
                    return 0
        print("\n❌ Все методы скачивания не удались")
        return 1
    else:
        # Используем указанный метод
        if method in methods:
            method_name, method_func = methods[method]
            if method_func():
                if check_downloaded_files():
                    return 0
            return 1
        else:
            print(f"❌ Неизвестный метод: {method}")
            print(f"Доступные методы: {', '.join(methods.keys())}")
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скачивание Cross-Encoder модели")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "1", "2", "3"],
        help="Метод скачивания: auto (все по порядку), 1 (huggingface_hub), 2 (transformers), 3 (sentence_transformers)"
    )
    
    args = parser.parse_args()
    
    exit_code = download_model(args.method)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ Успешно! Модель готова к использованию.")
        print(f"Путь в .env: CROSS_ENCODER_MODEL={LOCAL_PATH}")
        print("=" * 60)
    
    sys.exit(exit_code)

