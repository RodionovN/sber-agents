"""
Скрипт для очистки блокировок HuggingFace Hub
Помогает при проблемах с зависшими загрузками
"""
import os
import sys
from pathlib import Path

def clear_hf_locks():
    """Очищает блокировки HuggingFace Hub"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    locks_dir = cache_dir / ".locks"
    
    print("=" * 60)
    print("Очистка блокировок HuggingFace Hub")
    print("=" * 60)
    
    if not locks_dir.exists():
        print("Папка блокировок не найдена")
        return 0
    
    # Находим блокировки для конкретной модели
    model_locks = locks_dir / "models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1"
    
    if model_locks.exists():
        print(f"\nНайдены блокировки для модели: {model_locks}")
        lock_files = list(model_locks.glob("*.lock"))
        
        if lock_files:
            print(f"Найдено блокировок: {len(lock_files)}")
            for lock_file in lock_files:
                try:
                    lock_file.unlink()
                    print(f"  ✓ Удалена блокировка: {lock_file.name}")
                except Exception as e:
                    print(f"  ✗ Не удалось удалить {lock_file.name}: {e}")
        else:
            print("Блокировки не найдены")
    else:
        print(f"\nБлокировки для модели не найдены: {model_locks}")
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    try:
        exit_code = clear_hf_locks()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

