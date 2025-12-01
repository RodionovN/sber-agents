#!/usr/bin/env python3
"""
Простой HTTP сервер для транскрибации аудио через faster-whisper.
Запускается на сервере где установлен faster-whisper.
"""
import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max file size

# Глобальная переменная для модели (загружается один раз)
whisper_model = None

def load_whisper_model():
    """Загружает модель Whisper один раз при старте."""
    global whisper_model
    if whisper_model is not None:
        return whisper_model
    
    try:
        from faster_whisper import WhisperModel
        
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        device = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Loading Whisper model: {model_size}, device: {device}, compute_type: {compute_type}")
        whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully")
        return whisper_model
    except ImportError:
        logger.error("faster-whisper not installed")
        raise
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}", exc_info=True)
        raise

@app.route('/health', methods=['GET'])
def health():
    """Проверка работоспособности сервера."""
    return jsonify({"status": "ok", "model_loaded": whisper_model is not None})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Транскрибация аудиофайла."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Загружаем модель если еще не загружена
        model = load_whisper_model()
        
        # Сохраняем файл во временную директорию
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Транскрибируем
            language = request.form.get('language', 'ru')
            logger.info(f"Transcribing file: {tmp_path}, language: {language}")
            
            segments, info = model.transcribe(tmp_path, language=language)
            
            # Собираем текст из всех сегментов
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text)
            
            result_text = " ".join(transcript_parts).strip()
            
            logger.info(f"Transcription successful: {len(result_text)} chars, language: {info.language}, probability: {info.language_probability:.2f}")
            
            return jsonify({
                "text": result_text,
                "language": info.language,
                "language_probability": info.language_probability
            })
        finally:
            # Удаляем временный файл
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")
    
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Загружаем модель при старте
    try:
        load_whisper_model()
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        logger.error("Server will start but transcription will fail")
    
    # Запускаем сервер
    host = os.getenv("WHISPER_API_HOST", "0.0.0.0")
    port = int(os.getenv("WHISPER_API_PORT", "8080"))
    logger.info(f"Starting Whisper API server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

