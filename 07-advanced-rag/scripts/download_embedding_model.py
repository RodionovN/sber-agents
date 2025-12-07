"""
Скрипт для скачивания HuggingFace embedding модели в локальную папку
Поддерживает несколько методов скачивания
"""
import os
import sys
import argparse
from pathlib import Path

# Отключаем использование Xet storage для избежания проблем с загрузкой
os.environ['HF_HUB_DISABLE_XET'] = '1'

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Модель по умолчанию из конфига
MODEL_NAME = "intfloat/multilingual-e5-base"
# Локальная папка для сохранения (можно изменить через аргумент)
DEFAULT_LOCAL_PATH = str(Path(project_root) / "models" / "multilingual-e5-base")


def method_huggingface_hub(local_path: str):
    """Метод 1: Использование huggingface_hub.snapshot_download (рекомендуется)"""
    print("=" * 60)
    print("Метод 1: Использование huggingface_hub.snapshot_download")
    print("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        
        # Отключаем использование Xet storage для избежания проблем
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        
        print(f"Скачивание модели {MODEL_NAME} в {local_path}...")
        os.makedirs(local_path, exist_ok=True)
        
        # Скачиваем только необходимые файлы для PyTorch (исключаем ONNX, OpenVINO)
        # Это уменьшает размер и избегает проблем с большими файлами
        ignore_patterns = ["*.onnx", "*.xml", "*.bin"]  # Игнорируем ONNX и OpenVINO файлы
        
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=local_path,
            ignore_patterns=["onnx/**", "openvino/**"],  # Игнорируем папки с ONNX и OpenVINO
            resume_download=True  # Продолжить скачивание, если было прервано
        )
        
        print(f"✅ Модель успешно скачана в {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при скачивании через huggingface_hub: {e}")
        import traceback
        traceback.print_exc()
        return False


def method_sentence_transformers(local_path: str):
    """Метод 2: Использование sentence_transformers.SentenceTransformer"""
    print("=" * 60)
    print("Метод 2: Использование sentence_transformers.SentenceTransformer")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Отключаем использование Xet storage для избежания проблем
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        
        print(f"Скачивание модели {MODEL_NAME}...")
        os.makedirs(local_path, exist_ok=True)
        
        # Загружаем модель из HuggingFace Hub
        print("Загрузка модели из HuggingFace Hub...")
        model = SentenceTransformer(MODEL_NAME)
        
        # Сохраняем модель в локальную папку
        print(f"Сохранение модели в {local_path}...")
        model.save(local_path)
        
        print(f"✅ Модель успешно скачана в {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при скачивании через sentence_transformers: {e}")
        import traceback
        traceback.print_exc()
        return False


def method_transformers(local_path: str):
    """Метод 3: Использование transformers.AutoModel"""
    print("=" * 60)
    print("Метод 3: Использование transformers.AutoModel")
    print("=" * 60)
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Отключаем использование Xet storage для избежания проблем
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        
        print(f"Скачивание модели {MODEL_NAME} в {local_path}...")
        os.makedirs(local_path, exist_ok=True)
        
        # Скачиваем модель и токенайзер
        print("Загрузка модели...")
        model = AutoModel.from_pretrained(MODEL_NAME)
        
        print("Загрузка токенайзера...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Сохраняем в локальную папку
        print(f"Сохранение в {local_path}...")
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        
        print(f"✅ Модель успешно скачана в {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при скачивании через transformers: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_downloaded_files(local_path: str):
    """Проверяет наличие скачанных файлов"""
    print("\n" + "=" * 60)
    print("Проверка скачанных файлов")
    print("=" * 60)
    
    required_files = [
        'config.json',
        'tokenizer_config.json',
        'vocab.txt',
        'sentence_bert_config.json'  # Для sentence-transformers
    ]
    
    model_files = ['pytorch_model.bin', 'model.safetensors']
    
    found_files = []
    found_model = False
    
    for f in required_files + model_files:
        file_path = os.path.join(local_path, f)
        if os.path.exists(file_path):
            found_files.append(f)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  ✓ {f} ({file_size:.2f} MB)")
            if f in model_files:
                found_model = True
    
    if found_model:
        print(f"\n✅ Модель готова к использованию!")
        print(f"Найдены файлы PyTorch модели")
        return True
    else:
        print("\n⚠️  Файлы модели PyTorch не найдены.")
        return False


def download_model(model_name: str = MODEL_NAME, local_path: str = DEFAULT_LOCAL_PATH, method: str = "auto"):
    """Скачивает модель используя указанный метод"""
    print(f"\nСкачивание модели: {model_name}")
    print(f"Целевая папка: {local_path}")
    print(f"Размер модели: ~560MB\n")
    
    methods = {
        "1": ("huggingface_hub", method_huggingface_hub),
        "2": ("sentence_transformers", method_sentence_transformers),
        "3": ("transformers", method_transformers),
    }
    
    if method == "auto":
        # Пробуем все методы по порядку
        for method_name, method_func in methods.values():
            print(f"\nПробуем метод: {method_name}...")
            if method_func(local_path):
                if check_downloaded_files(local_path):
                    return 0
        print("\n❌ Все методы скачивания не удались")
        return 1
    else:
        # Используем указанный метод
        if method in methods:
            method_name, method_func = methods[method]
            if method_func(local_path):
                if check_downloaded_files(local_path):
                    return 0
            return 1
        else:
            print(f"❌ Неизвестный метод: {method}")
            print(f"Доступные методы: {', '.join(methods.keys())}")
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скачивание HuggingFace embedding модели")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Название модели на HuggingFace Hub (по умолчанию: {MODEL_NAME})"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_LOCAL_PATH,
        help=f"Локальная папка для сохранения (по умолчанию: {DEFAULT_LOCAL_PATH})"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "1", "2", "3"],
        help="Метод скачивания: auto (все по порядку), 1 (huggingface_hub), 2 (sentence_transformers), 3 (transformers)"
    )
    
    args = parser.parse_args()
    
    exit_code = download_model(args.model, args.path, args.method)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ Успешно! Модель готова к использованию.")
        print(f"\nДля использования локальной модели добавьте в .env:")
        print(f"HUGGINGFACE_EMBEDDING_MODEL={args.path}")
        print("=" * 60)
    
    sys.exit(exit_code)

