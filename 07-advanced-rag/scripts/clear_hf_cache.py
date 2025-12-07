"""
Скрипт для очистки кеша HuggingFace
Может помочь при ошибках Access Violation при загрузке моделей
"""
import sys
import os

def clear_hf_cache():
    """Очищает кеш HuggingFace"""
    try:
        from huggingface_hub import scan_cache_dir
        from pathlib import Path
        import shutil
        
        print("=" * 60)
        print("Очистка кеша HuggingFace")
        print("=" * 60)
        
        # Сканируем кеш
        print("\nСканирование кеша...")
        cache_info = scan_cache_dir()
        
        print(f"Найдено репозиториев: {len(cache_info.repos)}")
        print(f"Общий размер: {cache_info.size_on_disk_str}")
        
        if cache_info.repos:
            print("\nРепозитории в кеше:")
            for repo in cache_info.repos:
                print(f"  - {repo.repo_id} ({repo.size_on_disk_str})")
                for revision in repo.revisions:
                    print(f"    Ревизия: {revision.commit_hash[:8]} ({revision.size_on_disk_str})")
            
            # Определяем путь к кешу
            cache_path = Path.home() / ".cache" / "huggingface"
            if not cache_path.exists():
                # Пробуем найти через переменную окружения
                cache_path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            
            print("\n" + "=" * 60)
            print(f"Удаление кеша из: {cache_path}")
            print("=" * 60)
            
            if cache_path.exists():
                # Удаляем содержимое кеша
                for item in cache_path.iterdir():
                    if item.is_dir():
                        print(f"Удаление: {item.name}")
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        print(f"Удаление файла: {item.name}")
                        item.unlink(missing_ok=True)
                print("✅ Кеш очищен успешно!")
            else:
                print(f"⚠️  Путь к кешу не найден: {cache_path}")
                print("Попробуйте удалить вручную:")
                print(f"  {cache_path}")
        else:
            print("\nКеш пуст - нечего удалять")
        
        print("\n" + "=" * 60)
        print("Готово! Теперь попробуйте запустить бота снова.")
        print("=" * 60)
        
        return 0
        
    except ImportError:
        print("❌ Ошибка: huggingface_hub не установлен")
        print("Установите: uv pip install huggingface_hub")
        return 1
    except Exception as e:
        print(f"❌ Ошибка при очистке кеша: {e}")
        import traceback
        traceback.print_exc()
        return 1

def show_cache_info():
    """Показывает информацию о кеше без удаления"""
    try:
        from huggingface_hub import scan_cache_dir
        
        print("=" * 60)
        print("Информация о кеше HuggingFace")
        print("=" * 60)
        
        cache_info = scan_cache_dir()
        
        print(f"\nНайдено репозиториев: {len(cache_info.repos)}")
        print(f"Общий размер: {cache_info.size_on_disk_str}")
        
        if cache_info.repos:
            print("\nРепозитории в кеше:")
            for repo in cache_info.repos:
                print(f"  - {repo.repo_id} ({repo.size_on_disk_str})")
                for revision in repo.revisions:
                    print(f"    Ревизия: {revision.commit_hash[:8]} ({revision.size_on_disk_str})")
        else:
            print("\nКеш пуст")
        
        print("\n" + "=" * 60)
        
    except ImportError:
        print("❌ Ошибка: huggingface_hub не установлен")
        return 1
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        sys.exit(show_cache_info())
    else:
        print("Этот скрипт очистит весь кеш HuggingFace.")
        print("Для просмотра информации о кеше используйте: python scripts/clear_hf_cache.py --info")
        print("\nПродолжить? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response == 'y':
                sys.exit(clear_hf_cache())
            else:
                print("Отменено")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\nОтменено")
            sys.exit(0)

