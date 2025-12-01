import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from config import config

logger = logging.getLogger(__name__)

# Глобальное векторное хранилище
vector_store = None
retriever = None

# Кеши для промптов и LLM клиентов
_conversational_answering_prompt = None
_retrieval_query_transform_prompt = None
_llm_query_transform = None
_llm = None

def initialize_retriever():
    """Инициализация retriever из векторного хранилища"""
    global retriever
    if vector_store is None:
        logger.error("Cannot initialize retriever: vector_store is None")
        return False
    
    retriever = vector_store.as_retriever(search_kwargs={'k': config.RETRIEVER_K})
    logger.info(f"Retriever initialized with k={config.RETRIEVER_K}")
    return True

def format_chunks(chunks):
    """
    Форматирование чанков с метаданными для лучшей прозрачности
    """
    if not chunks:
        return "Нет доступной информации"
    
    formatted_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Получаем метаданные
        source = chunk.metadata.get('source', 'Unknown')
        page = chunk.metadata.get('page', 'N/A')
        doc_type = chunk.metadata.get('type', '')
        
        # Извлекаем имя файла из пути
        source_name = source.split('/')[-1] if '/' in source else source
        source_name = source_name.split('\\')[-1] if '\\' in source_name else source_name
        
        # Для JSON документов используем специальное форматирование
        if doc_type == 'json_qa':
            question = chunk.metadata.get('question', '')
            answer = chunk.metadata.get('answer', '')
            category = chunk.metadata.get('category', '')
            
            # Используем метаданные если они есть, иначе используем page_content
            if question and answer:
                formatted_text = f"[Источник {i}: База знаний Сбербанка"
                if category:
                    formatted_text += f", категория: {category}"
                formatted_text += "]\n"
                formatted_text += f"Вопрос: {question}\n\n"
                formatted_text += f"Ответ: {answer}"
            else:
                # Если метаданные отсутствуют, используем page_content
                formatted_text = f"[Источник {i}: База знаний Сбербанка"
                if category:
                    formatted_text += f", категория: {category}"
                formatted_text += "]\n"
                formatted_text += chunk.page_content
            
            formatted_parts.append(formatted_text)
        else:
            # Для PDF документов используем стандартное форматирование
            formatted_parts.append(
                f"[Источник {i}: {source_name}, стр. {page}]\n{chunk.page_content}"
            )
    
    return "\n\n---\n\n".join(formatted_parts)

def _load_prompts():
    """Ленивая загрузка промптов с обработкой ошибок"""
    global _conversational_answering_prompt, _retrieval_query_transform_prompt
    
    if _conversational_answering_prompt is not None:
        return _conversational_answering_prompt, _retrieval_query_transform_prompt
    
    try:
        conversation_system_text = config.load_prompt(config.CONVERSATION_SYSTEM_PROMPT_FILE)
        query_transform_text = config.load_prompt(config.QUERY_TRANSFORM_PROMPT_FILE)
        
        _conversational_answering_prompt = ChatPromptTemplate(
            [
                ("system", conversation_system_text),
                ("placeholder", "{messages}")
            ]
        )
        
        _retrieval_query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                ("user", query_transform_text),
            ]
        )
        
        logger.info("Prompts loaded successfully")
        return _conversational_answering_prompt, _retrieval_query_transform_prompt
        
    except FileNotFoundError as e:
        logger.error(f"Prompt file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompts: {e}", exc_info=True)
        raise

def _get_llm_query_transform():
    """Ленивая инициализация LLM для query transformation с кешированием"""
    global _llm_query_transform
    if _llm_query_transform is None:
        # Настройка заголовков для OpenRouter (если используется)
        headers = {}
        if config.OPENAI_BASE_URL and "openrouter.ai" in config.OPENAI_BASE_URL:
            headers["HTTP-Referer"] = "https://github.com/aidialogs/sber-agents"
            headers["X-Title"] = "Sber Agents RAG Bot"
        
        _llm_query_transform = ChatOpenAI(
            model=config.MODEL_QUERY_TRANSFORM,
            temperature=0.4,
            openai_api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            default_headers=headers if headers else None
        )
        logger.info(f"Query transform LLM initialized: {config.MODEL_QUERY_TRANSFORM}")
    return _llm_query_transform

def _get_llm():
    """Ленивая инициализация основной LLM с кешированием"""
    global _llm
    if _llm is None:
        # Настройка заголовков для OpenRouter (если используется)
        headers = {}
        if config.OPENAI_BASE_URL and "openrouter.ai" in config.OPENAI_BASE_URL:
            headers["HTTP-Referer"] = "https://github.com/aidialogs/sber-agents"
            headers["X-Title"] = "Sber Agents RAG Bot"
        
        _llm = ChatOpenAI(
            model=config.MODEL,
            temperature=0.9,
            openai_api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            default_headers=headers if headers else None
        )
        logger.info(f"Main LLM initialized: {config.MODEL}")
    return _llm

def get_retrieval_query_transformation_chain():
    """Цепочка трансформации запроса"""
    _, retrieval_query_transform_prompt = _load_prompts()
    return (
        retrieval_query_transform_prompt
        | _get_llm_query_transform()
        | StrOutputParser()
    )

def get_rag_chain():
    """Финальная RAG-цепочка с query transformation"""
    if retriever is None:
        raise ValueError("Retriever not initialized")
    
    conversational_answering_prompt, _ = _load_prompts()
    
    return (
        RunnablePassthrough.assign(
            context=get_retrieval_query_transformation_chain() | retriever | format_chunks
        )
        | conversational_answering_prompt
        | _get_llm()
        | StrOutputParser()
    )

async def rag_answer(messages):
    """
    Получить ответ от RAG с учетом истории диалога
    
    Args:
        messages: список LangChain messages (HumanMessage, AIMessage)
    
    Returns:
        str: ответ от RAG
    """
    if vector_store is None or retriever is None:
        logger.error("Vector store or retriever not initialized")
        raise ValueError("Векторное хранилище не инициализировано. Запустите индексацию.")
    
    # Трансформируем запрос для логирования
    _, retrieval_query_transform_prompt = _load_prompts()
    transformed_query = await (retrieval_query_transform_prompt | _get_llm_query_transform() | StrOutputParser()).ainvoke({"messages": messages})
    logger.info(f"Transformed query: {transformed_query}")
    
    # Получаем релевантные документы для логирования
    retrieved_docs = await retriever.ainvoke(transformed_query)
    
    # Логируем результаты поиска
    logger.info(f"Retrieved {len(retrieved_docs)} documents")
    json_count = sum(1 for doc in retrieved_docs if doc.metadata.get('type') == 'json_qa')
    pdf_count = len(retrieved_docs) - json_count
    logger.info(f"Sources breakdown: {json_count} JSON documents, {pdf_count} PDF documents")
    
    if json_count == 0 and pdf_count > 0:
        logger.warning("Warning: No JSON documents found in retrieval results, only PDF documents")
        # Логируем первые несколько источников для отладки
        for i, doc in enumerate(retrieved_docs[:3]):
            logger.debug(f"Retrieved doc {i+1}: type={doc.metadata.get('type', 'unknown')}, source={doc.metadata.get('source', 'unknown')}")
    
    # Форматируем контекст для логирования
    formatted_context = format_chunks(retrieved_docs)
    
    # Логируем первые 500 символов контекста для отладки
    if json_count > 0:
        logger.info(f"Context preview (first 500 chars): {formatted_context[:500]}...")
        # Проверяем, есть ли ответы в контексте
        if "Ответ:" in formatted_context:
            logger.info("Found 'Ответ:' in context - answers should be extractable")
        else:
            logger.warning("Warning: 'Ответ:' not found in formatted context")
    
    # Используем оригинальную RAG цепочку
    rag_chain = get_rag_chain()
    result = await rag_chain.ainvoke({"messages": messages})
    return result

def get_vector_store_stats():
    """Возвращает статистику векторного хранилища"""
    if vector_store is None:
        return {"status": "not initialized", "count": 0}
    
    doc_count = len(vector_store.store) if hasattr(vector_store, 'store') else 0
    return {"status": "initialized", "count": doc_count}

