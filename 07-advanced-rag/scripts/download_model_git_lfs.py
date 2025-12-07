"""
Скрипт для скачивания модели через Git LFS
Этот метод более надежен для больших файлов
"""
import os
import sys
import subprocess
from pathlib import Path

MODEL_NAME = "intfloat/multilingual-e5-base"
LOCAL_PATH = Path(r"D:\Documents\Projects\sber-agents\models\multilingual-e5-base")

def check_git_lfs():
    """Проверяет, установлен ли Git LFS"""
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✅ Git LFS установлен: {result.stdout.strip()}")
            return True
        else:
            print("❌ Git LFS не найден")
            return False
    except FileNotFoundError:
        print("❌ Git не установлен или не найден в PATH")
        return False
    except Exception as e:
        print(f"❌ Ошибка при проверке Git LFS: {e}")
        return False

def safe_remove_directory(path: Path):
    """Безопасное удаление директории в Windows"""
    import shutil
    import stat
    import platform
    
    def handle_remove_readonly(func, path, exc):
        """Обработчик для удаления файлов только для чтения"""
        os.chmod(path, stat.S_IWRITE)
        func(path)
    
    try:
        # Пробуем стандартный способ
        shutil.rmtree(path, onerror=handle_remove_readonly)
        return True
    except PermissionError:
        # Если не получилось, пробуем через PowerShell в Windows
        if platform.system() == "Windows":
            try:
                import subprocess
                print("Пробуем удалить через PowerShell...")
                result = subprocess.run(
                    ["powershell", "-Command", f"Remove-Item -Path '{path}' -Recurse -Force"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    return True
            except Exception as e:
                print(f"⚠️  PowerShell также не помог: {e}")
        
        print(f"⚠️  Не удалось удалить папку: {path}")
        print("Попробуйте удалить вручную или использовать обновление существующего репозитория")
        print(f"Или выполните в PowerShell: Remove-Item -Path '{path}' -Recurse -Force")
        return False
    except Exception as e:
        print(f"⚠️  Не удалось удалить папку: {e}")
        print("Попробуйте удалить вручную или использовать обновление существующего репозитория")
        return False

def download_with_git_lfs():
    """Скачивает модель через Git LFS"""
    print("=" * 60)
    print(f"Скачивание модели {MODEL_NAME} через Git LFS")
    print(f"Целевая папка: {LOCAL_PATH}")
    print("=" * 60)
    
    repo_url = f"https://huggingface.co/{MODEL_NAME}"
    
    if LOCAL_PATH.exists():
        # Проверяем наличие файла модели
        model_file = LOCAL_PATH / "model.safetensors"
        if model_file.exists():
            size_mb = model_file.stat().st_size / 1024 / 1024
            if size_mb > 100:  # Проверяем, что файл достаточно большой
                print(f"\n✓ Модель уже существует: {size_mb:.2f} MB")
                print("Пропускаем скачивание. Модель готова к использованию.")
                return 0
        
        # Если модель не найдена или слишком мала
        print(f"\n⚠️  Папка {LOCAL_PATH} существует, но модель не найдена или неполная")
        
        # Проверяем, является ли это Git репозиторием
        git_dir = LOCAL_PATH / ".git"
        if git_dir.exists():
            print("Обнаружен Git репозиторий. Загружаем LFS файлы...")
            
            try:
                # Пробуем восстановить файлы, если checkout не удался
                print("Проверка состояния репозитория...")
                status_result = subprocess.run(
                    ["git", "status"],
                    cwd=str(LOCAL_PATH),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Загружаем LFS файлы с повторными попытками
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    print(f"Попытка загрузки LFS файлов {attempt}/{max_retries}...")
                    result = subprocess.run(
                        ["git", "lfs", "pull"],
                        cwd=str(LOCAL_PATH),
                        capture_output=True,
                        text=True,
                        timeout=3600
                    )
                    
                    if result.returncode == 0:
                        print("✅ LFS файлы загружены")
                        break
                    else:
                        if attempt < max_retries:
                            print(f"⚠️  Попытка {attempt} не удалась: {result.stderr[:200]}")
                            print(f"Повторяем через 5 секунд...")
                            import time
                            time.sleep(5)
                        else:
                            print(f"⚠️  Не удалось загрузить LFS файлы после {max_retries} попыток")
                            print(f"Последняя ошибка: {result.stderr[:500]}")
                
                # Проверяем снова
                if model_file.exists():
                    size_mb = model_file.stat().st_size / 1024 / 1024
                    if size_mb > 100:
                        print(f"✅ Файл модели найден: {size_mb:.2f} MB")
                        return 0
                    else:
                        print(f"⚠️  Файл модели найден, но размер подозрительно мал: {size_mb:.2f} MB")
                else:
                    print("⚠️  Файл модели не найден после загрузки LFS")
                
            except Exception as e:
                print(f"⚠️  Ошибка при загрузке LFS: {e}")
        
        # Если модель все еще не найдена, предлагаем перескачать
        # Проверяем аргументы командной строки
        force_remove = '--force' in sys.argv
        
        if not force_remove:
            response = input("Удалить существующую папку и скачать заново? (y/n): ")
        else:
            response = 'y'
        
        if response.lower() == 'y':
            if safe_remove_directory(LOCAL_PATH):
                print("Папка удалена")
            else:
                print("Не удалось удалить папку. Попробуйте удалить вручную.")
                return 1
        else:
            print("Используем существующую папку")
            if model_file.exists():
                size_mb = model_file.stat().st_size / 1024 / 1024
                print(f"⚠️  Файл модели найден, но размер подозрительно мал: {size_mb:.2f} MB")
                print("Попробуйте выполнить вручную:")
                print(f"  cd {LOCAL_PATH}")
                print("  git lfs pull")
            else:
                print("⚠️  Файл модели не найден. Попробуйте выполнить:")
                print(f"  cd {LOCAL_PATH}")
                print("  git lfs pull")
            return 1
    
    print(f"\nКлонирование репозитория: {repo_url}")
    print("Это может занять некоторое время (~1.1GB)...")
    
    try:
        # Клонируем репозиторий
        result = subprocess.run(
            ["git", "clone", repo_url, str(LOCAL_PATH)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 час таймаут
        )
        
        # Проверяем результат клонирования
        if result.returncode == 0 or "Clone succeeded" in result.stderr:
            print("✅ Репозиторий клонирован (возможно с предупреждениями)")
            
            # Если checkout не удался, пробуем восстановить
            if "checkout failed" in result.stderr or "smudge filter lfs failed" in result.stderr:
                print("⚠️  Checkout не удался, пробуем восстановить файлы...")
                
                # Пробуем git restore
                print("Попытка 1: git restore...")
                restore_result = subprocess.run(
                    ["git", "restore", "--source=HEAD", ":/"],
                    cwd=str(LOCAL_PATH),
                    capture_output=True,
                    text=True,
                    timeout=3600
                )
                
                if restore_result.returncode != 0:
                    print(f"⚠️  git restore не помог: {restore_result.stderr}")
            
            # Загружаем LFS файлы (может потребоваться несколько попыток)
            print("\nЗагрузка LFS файлов...")
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                print(f"Попытка {attempt}/{max_retries}...")
                result = subprocess.run(
                    ["git", "lfs", "pull"],
                    cwd=str(LOCAL_PATH),
                    capture_output=True,
                    text=True,
                    timeout=3600
                )
                
                if result.returncode == 0:
                    print("✅ LFS файлы загружены успешно")
                    break
                else:
                    if attempt < max_retries:
                        print(f"⚠️  Попытка {attempt} не удалась, повторяем через 5 секунд...")
                        import time
                        time.sleep(5)
                    else:
                        print(f"⚠️  Не удалось загрузить LFS файлы после {max_retries} попыток")
                        print(f"Последняя ошибка: {result.stderr}")
                        print("\nПопробуйте выполнить вручную:")
                        print(f"  cd {LOCAL_PATH}")
                        print("  git lfs pull")
            
            # Проверяем наличие основных файлов
            model_file = LOCAL_PATH / "model.safetensors"
            if model_file.exists():
                size_mb = model_file.stat().st_size / 1024 / 1024
                if size_mb > 100:
                    print(f"\n✅ Файл модели найден: {size_mb:.2f} MB")
                    return 0
                else:
                    print(f"\n⚠️  Файл модели найден, но размер подозрительно мал: {size_mb:.2f} MB")
                    print("Возможно, файл не полностью скачан")
                    return 1
            else:
                print("\n⚠️  Файл model.safetensors не найден")
                print("Попробуйте выполнить вручную:")
                print(f"  cd {LOCAL_PATH}")
                print("  git lfs pull")
                return 1
        else:
            print(f"❌ Ошибка при клонировании:")
            print(result.stderr)
            return 1
            
    except subprocess.TimeoutExpired:
        print("❌ Таймаут при клонировании (более 1 часа)")
        return 1
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Скачивание модели через Git LFS")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Принудительно удалить существующую папку и скачать заново"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Обновить существующий репозиторий вместо удаления"
    )
    args = parser.parse_args()
    
    if not check_git_lfs():
        print("\n" + "=" * 60)
        print("Установка Git LFS:")
        print("=" * 60)
        print("Windows:")
        print("  1. Скачайте с https://git-lfs.github.com/")
        print("  2. Установите и перезапустите терминал")
        print("\nLinux:")
        print("  sudo apt-get install git-lfs")
        print("  git lfs install")
        print("\nMac:")
        print("  brew install git-lfs")
        print("  git lfs install")
        sys.exit(1)
    
    # Модифицируем функцию для поддержки аргументов
    if args.force and LOCAL_PATH.exists():
        print("Принудительное удаление существующей папки...")
        if not safe_remove_directory(LOCAL_PATH):
            print("Не удалось удалить папку. Попробуйте удалить вручную.")
            sys.exit(1)
    
    exit_code = download_with_git_lfs()
    sys.exit(exit_code)

