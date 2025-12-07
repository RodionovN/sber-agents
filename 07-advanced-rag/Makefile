.PHONY: install run dataset dataset-upload evaluate download-cross-encoder

install:
	uv sync

run:
	uv run python src/bot.py

dataset:
	uv run python src/dataset_synthesizer.py --create

dataset-upload:
	uv run python src/dataset_synthesizer.py --upload

evaluate:
	uv run python src/evaluation.py

download-cross-encoder:
	uv run python scripts/download_cross_encoder.py

download-cross-encoder-hub:
	uv run python scripts/download_cross_encoder.py --method 1

download-cross-encoder-transformers:
	uv run python scripts/download_cross_encoder.py --method 2

download-cross-encoder-sentence:
	uv run python scripts/download_cross_encoder.py --method 3

download-embedding-model:
	uv run python scripts/download_embedding_model.py

download-embedding-model-hub:
	uv run python scripts/download_embedding_model.py --method 1

download-embedding-model-sentence:
	uv run python scripts/download_embedding_model.py --method 2

download-embedding-model-git:
	uv run python scripts/download_model_git_lfs.py

download-embedding-model-git-force:
	uv run python scripts/download_model_git_lfs.py --force

download-embedding-model-manual:
	uv run python scripts/download_model_manual.py

clear-hf-cache:
	uv run python scripts/clear_hf_cache.py

clear-hf-locks:
	uv run python scripts/clear_hf_locks.py

hf-cache-info:
	uv run python scripts/clear_hf_cache.py --info

