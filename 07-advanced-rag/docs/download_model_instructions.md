# Инструкция по скачиванию модели embeddings

Если при запуске бота возникают проблемы с автоматической загрузкой модели `intfloat/multilingual-e5-base`, используйте один из следующих способов:

## Способ 1: Использование скрипта проекта (рекомендуется)

```bash
# Через Makefile
make download-embedding-model

# Или напрямую
uv run python scripts/download_embedding_model.py
```

Скрипт автоматически попробует несколько методов скачивания.

## Способ 2: Использование huggingface-cli

Если у вас установлен `huggingface-cli`:

```bash
# Установка (если не установлен)
pip install huggingface_hub[cli]

# Скачивание модели
huggingface-cli download intfloat/multilingual-e5-base --local-dir ./models/multilingual-e5-base
```

## Способ 3: Ручное скачивание с HuggingFace Hub

1. Перейдите на страницу модели: https://huggingface.co/intfloat/multilingual-e5-base
2. Скачайте следующие файлы в папку `models/multilingual-e5-base/`:
   - `config.json`
   - `model.safetensors` (или `pytorch_model.bin`)
   - `tokenizer_config.json`
   - `vocab.txt` (или `sentencepiece.bpe.model`)
   - `special_tokens_map.json`

3. В `.env` укажите путь к локальной папке:
```bash
HUGGINGFACE_EMBEDDING_MODEL=./models/multilingual-e5-base
```

## Способ 4: Использование Git LFS (рекомендуется при проблемах с сетью)

Если у вас установлен Git LFS, это самый надежный способ для больших файлов:

```bash
# Использование скрипта проекта
uv run python scripts/download_model_git_lfs.py

# Или вручную:
git clone https://huggingface.co/intfloat/multilingual-e5-base D:\Documents\Projects\sber-agents\models\multilingual-e5-base
cd D:\Documents\Projects\sber-agents\models\multilingual-e5-base
git lfs pull
```

**Установка Git LFS:**
# Windows: https://git-lfs.github.com/
# Linux: sudo apt-get install git-lfs
# Mac: brew install git-lfs

# Клонирование репозитория с моделью
git lfs install
git clone https://huggingface.co/intfloat/multilingual-e5-base ./models/multilingual-e5-base
```

## Решение проблем

### Проблема: Таймауты при скачивании

Если возникают таймауты при скачивании:

1. **Проверьте интернет-соединение** - модель весит ~1.1GB
2. **Используйте VPN** - иногда помогает обойти проблемы с доступом к HuggingFace Hub
3. **Скачайте вручную** - используйте Способ 3 или 4
4. **Используйте альтернативную модель** - попробуйте модель меньшего размера:
   ```bash
   HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   ```

### Проблема: Ошибка "Xet storage"

Если появляется ошибка про Xet storage:

1. Установите переменную окружения перед запуском:
   ```bash
   # Windows PowerShell
   $env:HF_HUB_DISABLE_XET="1"
   
   # Windows CMD
   set HF_HUB_DISABLE_XET=1
   
   # Linux/Mac
   export HF_HUB_DISABLE_XET=1
   ```

2. Или добавьте в `.env`:
   ```bash
   HF_HUB_DISABLE_XET=1
   ```

## Использование локальной модели

После скачивания модели, укажите путь к локальной папке в `.env`:

```bash
# Абсолютный путь
HUGGINGFACE_EMBEDDING_MODEL=D:\Documents\Projects\sber-agents\07-advanced-rag\models\multilingual-e5-base

# Или относительный путь от корня проекта
HUGGINGFACE_EMBEDDING_MODEL=./models/multilingual-e5-base
```

## Альтернативные модели (если основная не скачивается)

Если модель `intfloat/multilingual-e5-base` не скачивается из-за проблем с сетью, используйте альтернативные модели меньшего размера:

### Рекомендуемые альтернативы:

1. **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** (~420MB)
   - Хорошее качество для многоязычных задач
   - Быстрее скачивается

2. **`intfloat/multilingual-e5-small`** (~278MB)
   - Меньшая версия той же модели
   - Хорошее качество при меньшем размере

3. **`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`** (~420MB)
   - Альтернатива с хорошим качеством

### Использование альтернативной модели:

Укажите название модели в `.env`:
```bash
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Или скачайте локально:
```bash
# Через Git LFS
git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 D:\Documents\Projects\sber-agents\models\paraphrase-multilingual-MiniLM-L12-v2

# Затем в .env:
HUGGINGFACE_EMBEDDING_MODEL=D:\Documents\Projects\sber-agents\models\paraphrase-multilingual-MiniLM-L12-v2
```

