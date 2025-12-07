"""
Альтернативный скрипт для скачивания модели с использованием прямых HTTP запросов
Обходит проблемы с Xet storage и таймаутами
"""
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Отключаем Xet storage
os.environ['HF_HUB_DISABLE_XET'] = '1'

MODEL_NAME = "intfloat/multilingual-e5-base"
LOCAL_PATH = Path(r"D:\Documents\Projects\sber-agents\models\multilingual-e5-base")

# Основные файлы, необходимые для работы модели
REQUIRED_FILES = [
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling/config.json",
    "model.safetensors",  # Основной файл модели (~1.1GB)
]

def download_file(url: str, filepath: Path, chunk_size: int = 8192):
    """Скачивает файл с прогресс-баром"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"❌ Ошибка при скачивании {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()  # Удаляем частично скачанный файл
        return False

def get_file_url(repo_id: str, filename: str, revision: str = "main"):
    """Получает прямую ссылку на файл"""
    base_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    return base_url

def download_model_files():
    """Скачивает необходимые файлы модели"""
    print("=" * 60)
    print(f"Скачивание модели {MODEL_NAME}")
    print(f"Целевая папка: {LOCAL_PATH}")
    print("=" * 60)
    
    LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    
    # Получаем информацию о репозитории для получения правильного revision
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_info = api.repo_info(repo_id=MODEL_NAME, repo_type="model")
        revision = repo_info.sha
        print(f"Используем revision: {revision[:8]}")
    except Exception as e:
        print(f"⚠️  Не удалось получить revision, используем 'main': {e}")
        revision = "main"
    
    downloaded = []
    failed = []
    
    for filename in REQUIRED_FILES:
        filepath = LOCAL_PATH / filename
        
        # Пропускаем уже скачанные файлы
        if filepath.exists():
            file_size = filepath.stat().st_size
            if file_size > 0:
                print(f"✓ {filename} уже существует ({file_size / 1024 / 1024:.2f} MB)")
                downloaded.append(filename)
                continue
        
        print(f"\nСкачивание: {filename}")
        url = get_file_url(MODEL_NAME, filename, revision)
        
        if download_file(url, filepath):
            downloaded.append(filename)
            print(f"✅ {filename} скачан успешно")
        else:
            failed.append(filename)
            print(f"❌ Не удалось скачать {filename}")
    
    print("\n" + "=" * 60)
    print("Результаты скачивания:")
    print("=" * 60)
    print(f"✅ Скачано: {len(downloaded)}/{len(REQUIRED_FILES)}")
    print(f"❌ Ошибок: {len(failed)}")
    
    if failed:
        print("\nНе удалось скачать:")
        for f in failed:
            print(f"  - {f}")
    
    # Проверяем наличие основного файла модели
    model_file = LOCAL_PATH / "model.safetensors"
    if model_file.exists() and model_file.stat().st_size > 100 * 1024 * 1024:  # > 100MB
        print("\n✅ Основной файл модели найден!")
        print(f"Размер: {model_file.stat().st_size / 1024 / 1024:.2f} MB")
        return 0
    else:
        print("\n⚠️  Основной файл модели отсутствует или слишком мал")
        print("Попробуйте скачать вручную:")
        print(f"  https://huggingface.co/{MODEL_NAME}/resolve/main/model.safetensors")
        return 1

if __name__ == "__main__":
    try:
        exit_code = download_model_files()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Скачивание прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

