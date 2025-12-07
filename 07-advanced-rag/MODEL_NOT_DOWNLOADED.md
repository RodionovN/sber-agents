# Проблема: Модель не скачивается

## Симптомы
- Ошибка: `ReadTimeoutError` или `ConnectionError` при загрузке модели
- Бот не может запуститься из-за отсутствия модели

## Причина
Модель `intfloat/multilingual-e5-base` (1.1GB) не может скачаться из-за проблем с сетью HuggingFace Hub.

## Решение 1: Использовать альтернативную модель (РЕКОМЕНДУЕТСЯ)

Используйте модель меньшего размера, которая скачивается быстрее:

1. Откройте файл `.env`
2. Найдите строку:
   ```bash
   HUGGINGFACE_EMBEDDING_MODEL=D:\Documents\Projects\sber-agents\models\multilingual-e5-base
   ```
3. Замените на:
   ```bash
   HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   ```
4. Сохраните файл и запустите бота - модель скачается автоматически (~420MB)

## Решение 2: Скачать модель вручную

Если вы хотите использовать оригинальную модель:

1. Откройте в браузере: https://huggingface.co/intfloat/multilingual-e5-base/tree/main
2. Скачайте файл `model.safetensors` (кнопка Download)
3. Сохраните в: `D:\Documents\Projects\sber-agents\models\multilingual-e5-base\`
4. Убедитесь, что в `.env` указан правильный путь:
   ```bash
   HUGGINGFACE_EMBEDDING_MODEL=D:\Documents\Projects\sber-agents\models\multilingual-e5-base
   ```

## Решение 3: Использовать Git LFS (если установлен)

```bash
cd D:\Documents\Projects\sber-agents\models
git clone https://huggingface.co/intfloat/multilingual-e5-base
cd multilingual-e5-base
git lfs pull
```

## Проверка

После скачивания проверьте наличие файлов:
```powershell
Test-Path "D:\Documents\Projects\sber-agents\models\multilingual-e5-base\model.safetensors"
Test-Path "D:\Documents\Projects\sber-agents\models\multilingual-e5-base\config.json"
```

Оба должны вернуть `True`.

## Рекомендация

**Используйте Решение 1** - альтернативная модель работает так же хорошо, но скачивается в 2.5 раза быстрее и надежнее.

