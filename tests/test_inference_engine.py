"""Tests unitarios para service/inference/inference_engine.py

Todos los tests siguen el patron AAA (Arrange, Act, Assert).
Los tests son completamente unitarios: no requieren descarga del modelo real,
se usan mocks de model y processor para evitar dependencias de red/GPU.

Comando Make asociado:
    make test-inference  ->  Ejecuta solo los tests de este modulo
    make test            ->  Ejecuta todos los tests del proyecto
"""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from service.inference.inference_engine import run_inference


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rgb_image():
    """Imagen PIL RGB de 64x64 pixeles."""
    return Image.new("RGB", (64, 64), color=(100, 150, 200))

@pytest.fixture
def valid_image_bytes(rgb_image):
    """Imagen RGB valida serializada como bytes PNG."""
    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    return buf.getvalue()

@pytest.fixture
def mock_processor():
    """Processor mock que retorna pixel_values tensor (1, 3, 224, 224)."""
    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    return processor

@pytest.fixture
def mock_model():
    """Modelo mock con logits (1, 2) y id2label {0: 'AI', 1: 'Real'}."""
    model = MagicMock()
    logits = torch.tensor([[2.0, 0.5]])  # AI gana
    model.return_value = SimpleNamespace(logits=logits)
    model.config.id2label = {0: "AI", 1: "Real"}
    return model


# ---------------------------------------------------------------------------
# Tests - Estructura del output (existentes, sin cambios)
# ---------------------------------------------------------------------------

class TestRunInferenceOutput:

    def test_retorna_dict(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result, dict)

    def test_tiene_clave_label(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "label" in result

    def test_tiene_clave_label_id(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "label_id" in result

    def test_tiene_clave_scores(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "scores" in result

    def test_scores_es_dict(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result["scores"], dict)

    def test_scores_suman_aproximadamente_uno(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 1e-4

    def test_scores_son_floats(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        for v in result["scores"].values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Tests - Valores del output (existentes, sin cambios)
# ---------------------------------------------------------------------------

class TestRunInferenceLabel:

    def test_label_es_string(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result["label"], str)

    def test_label_id_es_int(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result["label_id"], int)

    def test_label_esta_en_scores(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert result["label"] in result["scores"]

    def test_label_correcto_segun_logits(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert result["label"] == "AI"
        assert result["label_id"] == 0

    def test_label_real_cuando_logit_mayor(self, rgb_image, mock_processor):
        model = MagicMock()
        model.return_value = SimpleNamespace(logits=torch.tensor([[0.1, 3.0]]))
        model.config.id2label = {0: "AI", 1: "Real"}
        result = run_inference(rgb_image, model, mock_processor)
        assert result["label"] == "Real"
        assert result["label_id"] == 1

    def test_scores_contiene_todas_las_clases(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert set(result["scores"].keys()) == {"AI", "Real"}


# ---------------------------------------------------------------------------
# Tests - Entrada bytes (existentes, sin cambios)
# ---------------------------------------------------------------------------

class TestRunInferenceDesdeBytes:

    def test_acepta_bytes_png_validos(self, valid_image_bytes, mock_model, mock_processor):
        result = run_inference(valid_image_bytes, mock_model, mock_processor)
        assert "label" in result

    def test_bytes_invalidos_lanza_value_error(self, mock_model, mock_processor):
        with pytest.raises(ValueError, match="No se pudo decodificar los bytes"):
            run_inference(b"datos_invalidos", mock_model, mock_processor)


# ---------------------------------------------------------------------------
# Tests - Manejo de errores (existentes, sin cambios)
# ---------------------------------------------------------------------------

class TestRunInferenceErrores:

    def test_tipo_invalido_lanza_type_error(self, mock_model, mock_processor):
        with pytest.raises(TypeError, match="Se esperaba PIL.Image.Image o bytes"):
            run_inference("ruta/imagen.jpg", mock_model, mock_processor)

    def test_modelo_falla_lanza_runtime_error(self, rgb_image, mock_processor):
        broken_model = MagicMock(side_effect=Exception("CUDA out of memory"))
        broken_model.config.id2label = {0: "AI", 1: "Real"}
        with pytest.raises(RuntimeError, match="Error durante la inferencia del modelo"):
            run_inference(rgb_image, broken_model, mock_processor)

    def test_modelo_sin_logits_lanza_value_error(self, rgb_image, mock_processor):
        bad_model = MagicMock()
        bad_model.return_value = SimpleNamespace(hidden_states=torch.zeros(1, 10))
        bad_model.config.id2label = {0: "AI", 1: "Real"}
        with pytest.raises(ValueError, match="El modelo no retorno 'logits'"):
            run_inference(rgb_image, bad_model, mock_processor)


# ---------------------------------------------------------------------------
# Tests - Timing (NUEVOS - issue #11)
# ---------------------------------------------------------------------------

class TestRunInferenceTiming:

    def test_tiene_clave_timing(self, rgb_image, mock_model, mock_processor):
        # Arrange / Act
        result = run_inference(rgb_image, mock_model, mock_processor)
        # Assert
        assert "timing" in result

    def test_timing_es_dict(self, rgb_image, mock_model, mock_processor):
        # Arrange / Act
        result = run_inference(rgb_image, mock_model, mock_processor)
        # Assert
        assert isinstance(result["timing"], dict)

    def test_timing_tiene_preprocessing_ms(self, rgb_image, mock_model, mock_processor):
        # Arrange / Act
        result = run_inference(rgb_image, mock_model, mock_processor)
        # Assert
        assert "preprocessing_ms" in result["timing"]

    def test_timing_tiene_inference_ms(self, rgb_image, mock_model, mock_processor):
        # Arrange / Act
        result = run_inference(rgb_image, mock_model, mock_processor)
        # Assert
        assert "inference_ms" in result["timing"]

    def test_timing_tiene_total_ms(self, rgb_image, mock_model, mock_processor):
        # Arrange / Act
        result = run_inference(rgb_image, mock_model, mock_processor)
        # Assert
        assert "total_ms" in result["timing"]

    def test_timing_valores_son_floats(self, rgb_image, mock_model, mock_processor):
        # Arrange / Act
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        # Assert
        assert isinstance(timing["preprocessing_ms"], float)
        assert isinstance(timing["inference_ms"], float)
        assert isinstance(timing["total_ms"], float)

    def test_timing_valores_son_positivos(self, rgb_image, mock_model, mock_processor):
        # Arrange / Act
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        # Assert
        assert timing["preprocessing_ms"] >= 0.0
        assert timing["inference_ms"] >= 0.0
        assert timing["total_ms"] >= 0.0

    def test_total_ms_es_suma_de_partes(self, rgb_image, mock_model, mock_processor):
        # Arrange / Act
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        # Assert
        expected = round(timing["preprocessing_ms"] + timing["inference_ms"], 3)
        assert abs(timing["total_ms"] - expected) < 0.01

    def test_timing_tiene_3_decimales(self, rgb_image, mock_model, mock_processor):
        # Arrange / Act
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        # Assert
        for key, val in timing.items():
            assert round(val, 3) == val, f"{key} tiene mas de 3 decimales: {val}"