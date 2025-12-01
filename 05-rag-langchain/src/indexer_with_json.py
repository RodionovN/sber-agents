import json
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from config import config

logger = logging.getLogger(__name__)

def load_json_documents(json_file_path: str) -> list:
    """
    Загрузка документов из JSON файла с вопросами-ответами
    Каждая пара Q&A становится отдельным чанком с сохранением метаданных
    """
    
    json_path = Path(json_file_path)
    if not json_path.exists():
        logger.warning(f"JSON file {json_file_path} does not exist")
        return []
    
    # Загружаем JSON вручную для сохранения метаданных
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        # Формируем текст для индексации с улучшенным форматированием
        # Включаем вопрос, ответ и категорию для лучшего поиска
        question = item.get('question', '').strip()
        answer = item.get('answer', '').strip()
        category = item.get('category', '').strip()
        
        # Создаем более естественный текст для индексации
        # Это помогает эмбеддингам лучше находить релевантные документы
        text_parts = []
        if category:
            text_parts.append(f"Категория: {category}")
        if question:
            text_parts.append(f"Вопрос: {question}")
        if answer:
            text_parts.append(f"Ответ: {answer}")
        
        # Объединяем все части с переносами строк для лучшей читаемости
        text_content = "\n\n".join(text_parts)
        
        # Также добавляем вариант без префиксов для более естественного поиска
        # Это помогает находить документы по ключевым словам из вопроса и ответа
        if question and answer:
            text_content += f"\n\n{question} {answer}"
        
        # Создаем документ с метаданными
        doc = Document(
            page_content=text_content,
            metadata={
                'source': str(json_path),
                'type': 'json_qa',
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'category': item.get('category', ''),
                'url': item.get('url', ''),
                'page': 0  # Для совместимости с format_chunks
            }
        )
        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} Q&A pairs from JSON")
    return documents

def load_pdf_documents(data_dir: str) -> list:
    """Загрузка всех PDF документов из директории"""
    pages = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Directory {data_dir} does not exist")
        return pages
    
    pdf_files = list(data_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages.extend(loader.load())
        logger.info(f"Loaded {pdf_file.name}")
    
    return pages

def split_documents(pages: list) -> list:
    """Разбиение документов с учетом структуры"""
    # Сепараторы для банковских документов
    # Пробуем разбивать по: двойным переносам строк, одинарным, пробелам
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=[
            "\n\n\n",    # Тройной перенос - обычно разделы
            "\n\n",      # Двойной перенос - параграфы
            "\n",        # Одинарный перенос
            ". ",        # Конец предложения
            " ",         # Пробелы
            ""           # Символы
        ],
        keep_separator=True  # Сохраняем разделители для контекста
    )
    chunks = text_splitter.split_documents(pages)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks: list):
    """Создание векторного хранилища"""
    # Настройка заголовков для OpenRouter (если используется)
    headers = {}
    if config.OPENAI_BASE_URL and "openrouter.ai" in config.OPENAI_BASE_URL:
        headers["HTTP-Referer"] = "https://github.com/aidialogs/sber-agents"
        headers["X-Title"] = "Sber Agents RAG Bot"
    
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        default_headers=headers if headers else None
    )
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    logger.info(f"Created vector store with {len(chunks)} chunks")
    return vector_store

async def reindex_all():
    """Полная переиндексация всех документов (PDF + JSON)"""
    logger.info("Starting full reindexing...")
    
    try:
        # 1. Загружаем и обрабатываем PDF документы
        pdf_pages = load_pdf_documents(config.DATA_DIR)
        if not pdf_pages:
            logger.warning("No PDF documents found to index")
        
        pdf_chunks = split_documents(pdf_pages) if pdf_pages else []
        
        # 2. Загружаем JSON с вопросами-ответами
        json_file = f"{config.DATA_DIR}/sberbank_help_documents.json"
        json_chunks = load_json_documents(json_file)
        
        # 3. Объединяем все чанки
        all_chunks = pdf_chunks + json_chunks
        
        if not all_chunks:
            logger.warning("No documents found to index")
            return None
        
        logger.info(f"Total chunks to index: {len(all_chunks)} (PDF: {len(pdf_chunks)}, JSON: {len(json_chunks)})")
            
        # 4. Создаём векторное хранилище
        vector_store = create_vector_store(all_chunks)
        logger.info("Reindexing completed successfully")
        return vector_store
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return None