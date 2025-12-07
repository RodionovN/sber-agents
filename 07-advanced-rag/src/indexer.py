import logging
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from config import config

# Отключаем использование Xet storage для избежания проблем с загрузкой моделей
os.environ['HF_HUB_DISABLE_XET'] = '1'

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

def create_embeddings():
    """
    Фабрика для создания embeddings по провайдеру из конфига
    Поддерживает: openai, huggingface
    """
    provider = config.EMBEDDING_PROVIDER.lower()
    
    if provider == "openai":
        logger.info(f"Creating OpenAI embeddings: {config.EMBEDDING_MODEL}")
        return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    
    elif provider == "huggingface":
        model_path = config.HUGGINGFACE_EMBEDDING_MODEL
        logger.info(f"Creating HuggingFace embeddings: {model_path} on {config.HUGGINGFACE_DEVICE}")
        
        # Проверяем, является ли путь локальным
        model_path_obj = Path(model_path)
        is_local_path = False
        
        # Проверяем несколько условий для определения локального пути:
        # 1. Путь существует и является директорией
        # 2. ИЛИ путь содержит обратные слеши (Windows) или начинается с ./
        # 3. ИЛИ путь является абсолютным путем Windows (начинается с буквы диска)
        if model_path_obj.exists() and model_path_obj.is_dir():
            # Дополнительно проверяем наличие файла модели
            model_file = model_path_obj / "model.safetensors"
            pytorch_file = model_path_obj / "pytorch_model.bin"
            config_file = model_path_obj / "config.json"
            
            if model_file.exists() or pytorch_file.exists():
                if config_file.exists():
                    logger.info(f"✓ Local model found: {model_path}")
                    is_local_path = True
                    model_path = str(model_path_obj.resolve())
                else:
                    logger.warning(f"⚠️  Model files found but config.json missing in {model_path}")
            else:
                logger.warning(f"⚠️  Directory exists but model files not found in {model_path}")
                logger.warning(f"   Looking for: model.safetensors or pytorch_model.bin")
        elif "\\" in model_path or model_path.startswith("./") or model_path.startswith(".\\") or (len(model_path) > 1 and model_path[1] == ":"):
            # Похоже на локальный путь, но папка не существует
            logger.warning(f"⚠️  Local path specified but directory doesn't exist: {model_path}")
            logger.warning(f"   Will try to load from HuggingFace Hub instead")
        
        if not is_local_path:
            logger.info(f"Using HuggingFace Hub model: {model_path}")
        
        # Если это локальный путь, проверяем наличие файлов перед загрузкой
        if is_local_path:
            model_file = model_path_obj / "model.safetensors"
            pytorch_file = model_path_obj / "pytorch_model.bin"
            config_file = model_path_obj / "config.json"
            
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Config file not found: {config_file}\n"
                    f"Model directory exists but is incomplete. Please download the model:\n"
                    f"  make download-embedding-model-git"
                )
            
            if not (model_file.exists() or pytorch_file.exists()):
                raise FileNotFoundError(
                    f"Model file not found in {model_path}\n"
                    f"Looking for: model.safetensors or pytorch_model.bin\n"
                    f"Please download the model:\n"
                    f"  make download-embedding-model-git"
                )
            
            logger.info(f"✓ Model files verified, loading from local path...")
        
        # Используем sentence-transformers напрямую для избежания проблем с langchain_huggingface
        # Это обходит известные проблемы с Access Violation при загрузке некоторых моделей
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading model with sentence-transformers (direct method)...")
            model = SentenceTransformer(
                model_path,
                device=config.HUGGINGFACE_DEVICE
            )
            
            # Создаем обертку для совместимости с LangChain
            class SentenceTransformerEmbeddingsWrapper:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts):
                    """Embed список документов"""
                    return self.model.encode(texts, normalize_embeddings=True).tolist()
                
                def embed_query(self, text):
                    """Embed один запрос"""
                    return self.model.encode([text], normalize_embeddings=True)[0].tolist()
            
            logger.info("✓ Model loaded successfully with sentence-transformers")
            return SentenceTransformerEmbeddingsWrapper(model)
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Direct sentence-transformers loading failed: {e}")
            
            # Если это локальный путь и произошла ошибка, не пытаемся загрузить из Hub
            if is_local_path:
                logger.error("=" * 60)
                logger.error("❌ ОШИБКА: Не удалось загрузить локальную модель!")
                logger.error(f"Путь: {model_path}")
                logger.error("=" * 60)
                logger.error("Возможные причины:")
                logger.error("1. Модель не полностью скачана")
                logger.error("2. Файлы модели повреждены")
                logger.error("3. Неправильный путь к модели")
                logger.error("")
                logger.error("Решения:")
                logger.error(f"1. Проверьте наличие файла: {model_path_obj / 'model.safetensors'}")
                logger.error("2. Скачайте модель заново:")
                logger.error("   make download-embedding-model-git")
                logger.error("3. Или используйте альтернативную модель в .env:")
                logger.error("   HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                logger.error("=" * 60)
                raise RuntimeError(f"Failed to load local model from {model_path}. See logs above for details.") from e
            
            logger.info("Falling back to langchain_huggingface...")
            # Fallback на стандартный способ только если это не локальный путь
            return HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': config.HUGGINGFACE_DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Use 'openai' or 'huggingface'")

def create_vector_store(chunks: list):
    """Создание векторного хранилища"""
    embeddings = create_embeddings()
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    logger.info(f"Created vector store with {len(chunks)} chunks")
    return vector_store

async def reindex_all():
    """Полная переиндексация всех документов (PDF + JSON)
    
    Returns:
        tuple: (vector_store, chunks) для инициализации retriever
    """
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
            return None, []
        
        logger.info(f"Total chunks to index: {len(all_chunks)} (PDF: {len(pdf_chunks)}, JSON: {len(json_documents)})")
        
        vector_store = create_vector_store(all_chunks)
        logger.info("Reindexing completed successfully")
        
        # Возвращаем vector_store и chunks для BM25
        return vector_store, all_chunks
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None, []
    except Exception as e:
        logger.error(f"Error during reindexing: {e}", exc_info=True)
        return None, []

