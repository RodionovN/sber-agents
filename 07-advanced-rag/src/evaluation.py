import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from langsmith import Client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas import evaluate

# Отключаем использование Xet storage для избежания проблем с загрузкой моделей
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
    AnswerSimilarity,
    ContextRecall,
    ContextPrecision,
)
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from config import config
import rag

logger = logging.getLogger(__name__)

# Глобальные инициализированные метрики
_ragas_metrics = None
_ragas_run_config = None

class RateLimitedLLM:
    """
    Wrapper для LLM с задержкой между запросами для избежания rate limit
    Потокобезопасный для использования с RAGAS
    """
    def __init__(self, llm, delay_seconds: float = 4.0):
        object.__setattr__(self, '_llm', llm)
        object.__setattr__(self, 'delay_seconds', delay_seconds)
        object.__setattr__(self, 'last_request_time', 0)
        object.__setattr__(self, '_lock', threading.Lock())
    
    def __setattr__(self, name, value):
        """Переопределяем setattr для избежания конфликтов"""
        if name.startswith('_') or name in ('delay_seconds', 'last_request_time', '_lock'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._llm, name, value)
    
    def _wait_for_rate_limit(self):
        """Потокобезопасная задержка для соблюдения rate limit"""
        with self._lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.delay_seconds:
                sleep_time = self.delay_seconds - time_since_last_request
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            self.last_request_time = time.time()
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._wait_for_rate_limit()
        return self._llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
    
    def invoke(self, input, config=None, **kwargs):
        """Метод invoke для совместимости с LangChain"""
        self._wait_for_rate_limit()
        return self._llm.invoke(input, config=config, **kwargs)
    
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._wait_for_rate_limit()
        return self._llm._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
    
    def __getattr__(self, name):
        """Делегируем все остальные атрибуты и методы к обернутому LLM"""
        return getattr(self._llm, name)

def create_ragas_embeddings():
    """
    Фабрика для создания RAGAS embeddings по провайдеру из конфига
    Поддерживает: openai, huggingface
    """
    provider = config.RAGAS_EMBEDDING_PROVIDER.lower()
    
    if provider == "openai":
        logger.info(f"Creating RAGAS OpenAI embeddings: {config.RAGAS_EMBEDDING_MODEL}")
        embedding_kwargs = {"model": config.RAGAS_EMBEDDING_MODEL}
        # Добавляем base_url и api_key если они есть в конфиге
        if config.OPENAI_BASE_URL:
            embedding_kwargs["base_url"] = config.OPENAI_BASE_URL
        if config.OPENAI_API_KEY:
            embedding_kwargs["api_key"] = config.OPENAI_API_KEY
        return OpenAIEmbeddings(**embedding_kwargs)
    
    elif provider == "huggingface":
        model_path = config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL
        logger.info(f"Creating RAGAS HuggingFace embeddings: {model_path} on {config.RAGAS_HUGGINGFACE_DEVICE}")
        
        # Проверяем, является ли путь локальным
        model_path_obj = Path(model_path)
        if model_path_obj.exists() and model_path_obj.is_dir():
            logger.info(f"Using local model path: {model_path}")
            # Убеждаемся, что путь абсолютный
            model_path = str(model_path_obj.resolve())
        else:
            logger.info(f"Using HuggingFace Hub model: {model_path}")
        
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': config.RAGAS_HUGGINGFACE_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    else:
        raise ValueError(f"Unknown RAGAS embedding provider: {provider}. Use 'openai' or 'huggingface'")

def init_ragas_metrics():
    """
    Инициализация RAGAS метрик (один раз)
    
    По образцу референсного ноутбука (раздел 5.1)
    """
    global _ragas_metrics, _ragas_run_config
    
    if _ragas_metrics is not None:
        return _ragas_metrics, _ragas_run_config
    
    logger.info("Initializing RAGAS metrics...")
    
    # Предупреждение о rate limit для бесплатных моделей OpenRouter
    rate_limit_delay = None
    if config.OPENAI_BASE_URL and "openrouter" in config.OPENAI_BASE_URL.lower():
        if "free" in (config.RAGAS_LLM_MODEL or "").lower():
            # Лимит: 16 запросов в минуту = ~3.75 секунды между запросами
            # Используем 4 секунды для безопасности
            rate_limit_delay = 4.0
            logger.warning(
                f"⚠️  Используется бесплатная модель OpenRouter с лимитом 16 запросов/минуту. "
                f"Добавлена задержка {rate_limit_delay} сек между запросами. "
                f"Оценка может занять больше времени."
            )
    
    # Настройка LLM и embeddings для RAGAS (фиксированные модели для единообразной оценки)
    # Создаем LLM с настройками из конфига (base_url и api_key для OpenRouter и других провайдеров)
    llm_kwargs = {
        "model": config.RAGAS_LLM_MODEL,
        "temperature": 0,
        "max_tokens": 4000  # Ограничение для бесплатного аккаунта OpenRouter
    }
    if config.OPENAI_BASE_URL:
        llm_kwargs["base_url"] = config.OPENAI_BASE_URL
    if config.OPENAI_API_KEY:
        llm_kwargs["api_key"] = config.OPENAI_API_KEY
    
    base_llm = ChatOpenAI(**llm_kwargs)
    
    # Обертываем LLM в rate limiter если нужно
    if rate_limit_delay:
        langchain_llm = RateLimitedLLM(base_llm, delay_seconds=rate_limit_delay)
        logger.info(f"✓ Rate limiting enabled: {rate_limit_delay}s delay between requests")
    else:
        langchain_llm = base_llm
    
    langchain_embeddings = create_ragas_embeddings()
    
    # Создаем метрики
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        AnswerCorrectness(),
        AnswerSimilarity(),
        ContextRecall(),
        ContextPrecision(),
    ]
    
    # Инициализируем метрики
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = ragas_llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = ragas_embeddings
        run_config = RunConfig()
        metric.init(run_config)
    
    # Настройки для выполнения
    # Уменьшаем параллелизм для избежания rate limit на бесплатных моделях OpenRouter
    # Лимит: 16 запросов в минуту для free моделей
    run_config = RunConfig(
        max_workers=1,  # Уменьшено с 4 до 1 для избежания rate limit
        max_wait=300,   # Увеличено время ожидания
        max_retries=5   # Больше попыток при rate limit
    )
    
    _ragas_metrics = metrics
    _ragas_run_config = run_config
    
    logger.info(f"✓ RAGAS metrics initialized: {', '.join([m.name for m in metrics])}")
    logger.info(f"✓ RAGAS LLM: {config.RAGAS_LLM_MODEL}")
    logger.info(f"✓ RAGAS Embedding Provider: {config.RAGAS_EMBEDDING_PROVIDER}")
    if config.RAGAS_EMBEDDING_PROVIDER == "openai":
        logger.info(f"✓ RAGAS Embedding Model: {config.RAGAS_EMBEDDING_MODEL}")
    else:
        logger.info(f"✓ RAGAS Embedding Model: {config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL} on {config.RAGAS_HUGGINGFACE_DEVICE}")
    
    return _ragas_metrics, _ragas_run_config

def check_dataset_exists(dataset_name: str) -> bool:
    """
    Проверка существования датасета в LangSmith
    
    Args:
        dataset_name: имя датасета
    
    Returns:
        True если датасет существует
    """
    if not config.LANGSMITH_API_KEY:
        logger.error("LANGSMITH_API_KEY not set")
        return False
    
    try:
        client = Client()
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        return len(datasets) > 0
    except Exception as e:
        logger.error(f"Error checking dataset: {e}")
        return False

def evaluate_dataset(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Главная функция evaluation RAG системы
    
    По образцу референсного ноутбука (раздел 5.2):
    1. Запуск эксперимента в LangSmith с blocking=False и сбор данных
    2. RAGAS batch evaluation
    3. Загрузка метрик как feedback в LangSmith
    
    Args:
        dataset_name: имя датасета (по умолчанию из конфига)
    
    Returns:
        dict с результатами evaluation
    """
    if not config.LANGSMITH_API_KEY:
        raise ValueError("LANGSMITH_API_KEY not set. Cannot run evaluation.")
    
    if dataset_name is None:
        dataset_name = config.LANGSMITH_DATASET
    
    logger.info(f"Starting evaluation for dataset: {dataset_name}")
    
    # Проверяем существование датасета
    if not check_dataset_exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' not found in LangSmith")
    
    # Инициализируем метрики
    ragas_metrics, ragas_run_config = init_ragas_metrics()
    
    client = Client()
    
    # ========== Шаг 1: Запуск эксперимента и сбор данных ==========
    logger.info("\n[1/3] Running experiment and collecting data...")
    
    # Создаем target функцию для нашего RAG
    def target(inputs: dict) -> dict:
        """Target функция для evaluation"""
        question = inputs["question"]
        
        # Используем существующую RAG цепочку
        # Передаем только вопрос (без истории для evaluation)
        from langchain_core.messages import HumanMessage
        result = rag.get_rag_chain().invoke({"messages": [HumanMessage(content=question)]})
        
        return {
            "answer": result["answer"],
            "documents": result["documents"]
        }
    
    # Собираем данные во время выполнения evaluate
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    run_ids = []
    
    # evaluate() с blocking=False возвращает итератор
    for result in client.evaluate(
        target,
        data=dataset_name,
        evaluators=[],
        experiment_prefix="rag-evaluation",
        metadata={
            "approach": "RAGAS batch evaluation + LangSmith feedback",
            "model": config.MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        blocking=False,
    ):
        run = result["run"]
        example = result["example"]
        
        # Получаем данные
        question = run.inputs.get("question", "")
        answer = run.outputs.get("answer", "")
        documents = run.outputs.get("documents", [])
        contexts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        ground_truth = example.outputs.get("answer", "") if example else ""
        
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)
        run_ids.append(str(run.id))
    
    logger.info(f"Experiment completed, collected {len(questions)} examples")
    
    # ========== Шаг 2: RAGAS evaluation ==========
    logger.info("\n[2/3] Running RAGAS evaluation...")
    
    # Создаем Dataset для RAGAS
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })
    
    # Запускаем evaluation
    ragas_result = evaluate(
        ragas_dataset,
        metrics=ragas_metrics,
        run_config=ragas_run_config,
    )
    
    ragas_df = ragas_result.to_pandas()
    
    logger.info("RAGAS evaluation completed")
    
    # Вычисляем средние значения метрик
    metrics_summary = {}
    for metric in ragas_metrics:
        if metric.name in ragas_df.columns:
            # Игнорируем nan значения при вычислении среднего
            valid_scores = ragas_df[metric.name].dropna()
            if len(valid_scores) > 0:
                avg_score = valid_scores.mean()
                metrics_summary[metric.name] = avg_score
                logger.info(f"  {metric.name}: {avg_score:.3f} (valid: {len(valid_scores)}/{len(ragas_df)})")
            else:
                metrics_summary[metric.name] = float('nan')
                logger.warning(f"  {metric.name}: nan (no valid scores - возможно rate limit или ошибки API)")
    
    # ========== Шаг 3: Загрузка feedback в LangSmith ==========
    logger.info("\n[3/3] Uploading feedback to LangSmith...")
    
    for idx, run_id in enumerate(run_ids):
        row = ragas_df.iloc[idx]
        
        for metric in ragas_metrics:
            if metric.name in row:
                score = row[metric.name]
                # Пропускаем nan значения
                if isinstance(score, float) and (score != score):  # Проверка на nan
                    logger.warning(f"Skipping nan score for {metric.name} in run {run_id}")
                    continue
                try:
                    client.create_feedback(
                        run_id=run_id,
                        key=metric.name,
                        score=float(score),
                        comment=f"RAGAS metric: {metric.name}"
                    )
                except Exception as e:
                    logger.error(f"Error creating feedback for {metric.name}: {e}")
    
    logger.info(f"Feedback uploaded ({len(run_ids)} runs)")
    
    return {
        "dataset_name": dataset_name,
        "num_examples": len(questions),
        "metrics": metrics_summary,
        "ragas_result": ragas_result,
        "run_ids": run_ids
    }

def main():
    """Main CLI function for evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG evaluation using RAGAS metrics")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (default: from config)")
    args = parser.parse_args()
    
    try:
        result = evaluate_dataset(args.dataset)
        
        # Выводим результаты
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Dataset: {result['dataset_name']}")
        print(f"Examples processed: {result['num_examples']}")
        print("\nRAGAS Metrics:")
        
        metric_descriptions = {
            "faithfulness": "Обоснованность (нет галлюцинаций)",
            "answer_relevancy": "Релевантность ответа",
            "answer_correctness": "Правильность ответа",
            "answer_similarity": "Похожесть на эталон",
            "context_recall": "Полнота контекста",
            "context_precision": "Точность поиска"
        }
        
        for metric_name, score in result["metrics"].items():
            desc = metric_descriptions.get(metric_name, metric_name)
            if isinstance(score, float) and not (score != score):  # Проверка на nan
                emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                print(f"{emoji} {desc}: {score:.3f}")
            else:
                print(f"🔴 {desc}: nan (ошибки при вычислении)")
        
        print("\n" + "=" * 70)
        print("Results uploaded to LangSmith as feedback")
        print("=" * 70)
        
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

