import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from openai import APIStatusError, APIError
from config import config

logger = logging.getLogger(__name__)

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
    """Разбиение документов на чанки"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(pages)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def load_json_documents(json_file_path: str) -> list:
    """Загрузка Q&A пар из JSON, каждая пара - отдельный чанк"""
    json_path = Path(json_file_path)
    if not json_path.exists():
        logger.warning(f"JSON file {json_file_path} does not exist")
        return []
    
    try:
        loader = JSONLoader(
            file_path=str(json_path),
            jq_schema='.[].full_text',
            text_content=False
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} Q&A pairs from JSON")
        return documents
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return []

def create_vector_store(chunks: list):
    """Создание векторного хранилища"""
    
    # Выбор embeddings на основе провайдера
    if config.EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        
        embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        logger.info(f"Using Ollama embeddings: {config.EMBEDDING_MODEL} (base_url: {config.OLLAMA_BASE_URL})")
    elif config.EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        from pathlib import Path
        
        # Проверяем, является ли путь локальным
        model_path = Path(config.EMBEDDING_MODEL)
        if model_path.exists() and model_path.is_dir():
            # Проверяем наличие файлов модели HuggingFace
            has_model_files = any(
                (model_path / f).exists() 
                for f in ['config.json', 'pytorch_model.bin', 'model.safetensors', 'tokenizer.json']
            )
            if has_model_files:
                # Используем локальный путь напрямую
                model_name = str(model_path.absolute())
                logger.info(f"Using local HuggingFace model: {model_name}")
            else:
                # Путь существует, но нет файлов модели - возможно это Ollama модель
                logger.warning(
                    f"Directory exists but no HuggingFace model files found: {model_path}\n"
                    f"This might be an Ollama model. Consider using EMBEDDING_PROVIDER=ollama instead."
                )
                raise ValueError(
                    f"No HuggingFace model files found in {model_path}. "
                    f"If this is an Ollama model, set EMBEDDING_PROVIDER=ollama"
                )
        else:
            # Используем имя модели из HuggingFace Hub
            model_name = config.EMBEDDING_MODEL
            logger.info(f"Using HuggingFace Hub model: {model_name}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # или 'cuda' если есть GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Using HuggingFace embeddings: {model_name}")
    else:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL
        )
        logger.info(f"Using OpenAI embeddings: {config.EMBEDDING_MODEL}")
    
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
        # Загрузка PDF документов
        pages = load_pdf_documents(config.DATA_DIR)
        pdf_chunks = split_documents(pages) if pages else []
        logger.info(f"PDF: {len(pdf_chunks)} chunks")
        
        # Загрузка JSON Q&A пар
        json_file = Path(config.DATA_DIR) / "sberbank_help_documents.json"
        json_documents = load_json_documents(str(json_file))
        logger.info(f"JSON: {len(json_documents)} Q&A pairs")
        
        # Объединяем все чанки
        all_chunks = pdf_chunks + json_documents
        
        if not all_chunks:
            logger.warning("No documents found to index")
            return None
        
        logger.info(f"Total chunks to index: {len(all_chunks)} (PDF: {len(pdf_chunks)}, JSON: {len(json_documents)})")
        
        vector_store = create_vector_store(all_chunks)
        logger.info("Reindexing completed successfully")
        return vector_store
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except APIStatusError as e:
        # Обработка ошибок API (например, недостаточно кредитов)
        error_code = getattr(e, 'status_code', None)
        error_message = str(e)
        
        # Попытка извлечь код из сообщения об ошибке, если status_code недоступен
        if error_code is None and "Error code:" in error_message:
            try:
                # Извлекаем код из сообщения вида "Error code: 402"
                error_code = int(error_message.split("Error code:")[1].split()[0])
            except (IndexError, ValueError):
                pass
        
        if error_code == 402:
            logger.error(
                "Insufficient API credits. Please check your API key and account balance. "
                f"Error details: {error_message}"
            )
        elif error_code == 401:
            logger.error(
                "Invalid API key. Please check your OPENAI_API_KEY configuration. "
                f"Error details: {error_message}"
            )
        elif error_code == 429:
            logger.error(
                "Rate limit exceeded. Please try again later. "
                f"Error details: {error_message}"
            )
        else:
            logger.error(
                f"API error (code {error_code}) during reindexing: {error_message}",
                exc_info=True
            )
        return None
    except APIError as e:
        # Общая обработка ошибок OpenAI API
        logger.error(f"OpenAI API error during reindexing: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return None

