# Быстрое решение проблемы со скачиванием модели

## Проблема
Модель `intfloat/multilingual-e5-base` не скачивается из-за проблем с сетью HuggingFace (таймауты, Xet storage).

## Решение 1: Использовать альтернативную модель (РЕКОМЕНДУЕТСЯ)

Используйте модель меньшего размера, которая скачивается быстрее:

1. Обновите `.env` файл:
```bash
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

2. Запустите бота - модель скачается автоматически (~420MB вместо 1.1GB)

## Решение 2: Скачать через Git LFS

Если у вас установлен Git LFS:

```bash
# Проверьте установку
git lfs version

# Если установлен, выполните:
make download-embedding-model-git

# Или вручную:
git clone https://huggingface.co/intfloat/multilingual-e5-base D:\Documents\Projects\sber-agents\models\multilingual-e5-base
cd D:\Documents\Projects\sber-agents\models\multilingual-e5-base
git lfs pull
```

## Решение 3: Скачать вручную через браузер

1. Откройте: https://huggingface.co/intfloat/multilingual-e5-base/tree/main
2. Скачайте файл `model.safetensors` (кнопка Download)
3. Сохраните в: `D:\Documents\Projects\sber-agents\models\multilingual-e5-base\`
4. Убедитесь, что в `.env` указан правильный путь:
   ```bash
   HUGGINGFACE_EMBEDDING_MODEL=D:\Documents\Projects\sber-agents\models\multilingual-e5-base
   ```

## Решение 4: Использовать VPN или другое интернет-соединение

Иногда проблемы с сетью решаются через VPN или другое подключение.

---

**Рекомендация:** Используйте Решение 1 (альтернативную модель) - это самый быстрый способ начать работу.

