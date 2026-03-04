.PHONY: test-preprocess help

help:
	@echo "Available commands:"
	@echo "  make test-preprocess    - Run image preprocessing test locally (requires HF_MODEL_ID env var)"

test-preprocess:
	@echo "Running preprocessing test..."
	python -m service.inference.test_preprocess_local