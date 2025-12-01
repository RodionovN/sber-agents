# Использование Ollama для Embeddings

## Установка Ollama

### Windows
1. Скачайте установщик: https://ollama.com/download/windows
2. Запустите установщик и следуйте инструкциям
3. Перезапустите терминал

### macOS
```bash
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Загрузка модели через Ollama

### Важно: Модель `intfloat/multilingual-e5-base` недоступна напрямую в Ollama

В реестре Ollama доступны альтернативные модели для embeddings:

1. **jeffh/intfloat-multilingual-e5-large** (рекомендуется, похожа на multilingual-e5-base)
2. **nomic-embed-text** (универсальная модель)
3. **all-minilm** (легкая модель)

### Способ 1: Использовать скрипт

```bash
# Загрузить похожую модель (рекомендуется)
uv run python scripts/download_model_ollama.py jeffh/intfloat-multilingual-e5-large

# Или использовать универсальную модель
uv run python scripts/download_model_ollama.py nomic-embed-text
```

### Способ 2: Использовать Ollama CLI напрямую

```bash
# Загрузить модель
ollama pull jeffh/intfloat-multilingual-e5-large

# Проверить, что модель загружена
ollama show jeffh/intfloat-multilingual-e5-large

# Протестировать embeddings
ollama run jeffh/intfloat-multilingual-e5-large "Тестовый текст"
```

## Интеграция Ollama в проект

### 1. Установить зависимости

Добавьте в `pyproject.toml`:
```toml
"langchain-ollama>=0.1.0",
```

Затем установите:
```bash
uv sync
```

### 2. Настроить `.env`

Добавьте в `.env`:
```bash
# Использовать Ollama для embeddings
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=jeffh/intfloat-multilingual-e5-large

# URL Ollama (по умолчанию http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Обновить код для поддержки Ollama

Код в `src/indexer.py` нужно обновить для поддержки Ollama embeddings (см. пример ниже).

## Проверка работы Ollama

### Проверить, что Ollama запущен:
```bash
ollama serve
```

### Проверить список загруженных моделей:
```bash
ollama list
```

### Протестировать embeddings через API:
```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "jeffh/intfloat-multilingual-e5-large",
  "prompt": "Тестовый текст для проверки embeddings"
}'
```

## Пример использования OllamaEmbeddings в коде

```python
from langchain_ollama import OllamaEmbeddings

# Создание embeddings через Ollama
embeddings = OllamaEmbeddings(
    model="jeffh/intfloat-multilingual-e5-large",
    base_url="http://localhost:11434"  # По умолчанию
)

# Использование в векторном хранилище
vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

## Преимущества использования Ollama

1. **Локальное выполнение** - не требует API ключей и интернета после загрузки модели
2. **Бесплатно** - нет лимитов и платы за использование
3. **Приватность** - данные не отправляются на внешние серверы
4. **Быстро** - работает локально без задержек сети

## Недостатки

1. **Требует ресурсов** - модели занимают место на диске и используют RAM/GPU
2. **Ограниченный выбор моделей** - не все модели HuggingFace доступны в Ollama
3. **Требует установки** - нужно установить и запустить Ollama сервер

