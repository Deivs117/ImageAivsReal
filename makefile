.PHONY: help install clean clean-grpc grpc test gui

# Detectar SO
UNAME := $(shell uname)

ifeq ($(OS),Windows_NT)
    # Windows
    RM_DIR := rmdir /s /q
    MKDIR := mkdir
else
    # Linux / Mac
    RM_DIR := rm -rf
    MKDIR := mkdir -p
endif

# ============================================
# TARGETS GENERALES
# ============================================

help:
	@echo "Comandos disponibles:"
	@echo "  make install      - Instalar dependencias (uv sync)"
	@echo "  make gui          - Ejecutar Streamlit GUI (app/app.py)"
	@echo "  make grpc         - Generar stubs gRPC desde proto/inference.proto"
	@echo "  make clean-grpc   - Limpiar stubs gRPC generados"
	@echo "  make clean        - Limpiar todo (__pycache__, .pytest_cache, proto/generated)"
	@echo "  make test         - Ejecutar tests con pytest"

# ============================================
# INSTALACIÓN Y DEPENDENCIAS
# ============================================

install:
	@echo "Installing dependencies with uv..."
	uv sync

# ============================================
# GUI (Streamlit)
# ============================================

gui:
	uv run -m streamlit run app/app.py

# ============================================
# gRPC TARGETS (Multiplataforma)
# ============================================

grpc:
	@echo "Creating proto/generated directory if it doesn't exist..."
ifeq ($(OS),Windows_NT)
	@if not exist proto\generated mkdir proto\generated
else
	@mkdir -p proto/generated
endif
	@echo "Generating gRPC stubs from proto/inference.proto..."
	uv run -m grpc_tools.protoc -I proto --python_out=proto/generated --grpc_python_out=proto/generated proto/inference.proto
	@echo "gRPC stubs generated successfully in proto/generated/"

clean-grpc:
	@echo "Removing gRPC generated stubs..."
ifeq ($(OS),Windows_NT)
	@if exist proto\generated $(RM_DIR) proto\generated
	@echo "gRPC stubs removed."
else
	@$(RM_DIR) proto/generated
	@echo "gRPC stubs removed."
endif

# ============================================
# LIMPIEZA
# ============================================

clean:
	@echo "Cleaning Python cache files..."
ifeq ($(OS),Windows_NT)
	@if exist __pycache__ $(RM_DIR) __pycache__
	@if exist .pytest_cache $(RM_DIR) .pytest_cache
	@if exist proto\generated $(RM_DIR) proto\generated
	@echo "Clean completed."
else
	@$(RM_DIR) __pycache__ .pytest_cache proto/generated 2>/dev/null || true
	@echo "Clean completed."
endif

# ============================================
# TESTS
# ============================================

test:
	@echo "Running tests with pytest..."
	uv run -m pytest tests/ -v

test-coverage:
	@echo "Running tests with coverage..."
	uv run -m pytest tests/ --cov --cov-report=html