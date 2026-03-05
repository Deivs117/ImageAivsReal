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
# Tests - Estructura del output exitoso (actualizados - issue #12)
# ---------------------------------------------------------------------------

class TestRunInferenceOutput:

    def test_retorna_dict(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result, dict)

    def test_tiene_clave_status(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "status" in result

    def test_status_ok_en_caso_exitoso(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert result["status"] == "ok"

    def test_tiene_clave_label(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "label" in result

    def test_tiene_clave_label_id(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "label_id" in result

    def test_tiene_clave_scores(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "scores" in result

    def test_tiene_clave_error(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "error" in result

    def test_error_es_none_en_caso_exitoso(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert result["error"] is None

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
# Tests - Valores del output
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
# Tests - Entrada bytes
# ---------------------------------------------------------------------------

class TestRunInferenceDesdeBytes:

    def test_acepta_bytes_png_validos(self, valid_image_bytes, mock_model, mock_processor):
        result = run_inference(valid_image_bytes, mock_model, mock_processor)
        assert result["status"] == "ok"
        assert "label" in result

    def test_bytes_invalidos_retorna_error_controlado(self, mock_model, mock_processor):
        result = run_inference(b"datos_invalidos", mock_model, mock_processor)
        assert result["status"] == "error"
        assert result["error"]["code"] == "INVALID_IMAGE"
        assert result["label"] is None
        assert result["scores"] == {{}}


# ---------------------------------------------------------------------------
# Tests - Manejo de errores controlados (actualizados - issue #12)
# ---------------------------------------------------------------------------

class TestRunInferenceErroresControlados:

    def test_tipo_invalido_retorna_error_controlado(self, mock_model, mock_processor):
        result = run_inference("ruta/imagen.jpg", mock_model, mock_processor)
        assert result["status"] == "error"
        assert result["error"]["code"] == "INVALID_IMAGE"
        assert result["label"] is None

    def test_modelo_falla_retorna_error_controlado(self, rgb_image, mock_processor):
        broken_model = MagicMock(side_effect=Exception("CUDA out of memory"))
        broken_model.config.id2label = {0: "AI", 1: "Real"}
        result = run_inference(rgb_image, broken_model, mock_processor)
        assert result["status"] == "error"
        assert result["error"]["code"] == "INFERENCE_ERROR"
        assert "CUDA out of memory" in result["error"]["message"]

    def test_modelo_sin_logits_retorna_error_controlado(self, rgb_image, mock_processor):
        bad_model = MagicMock()
        bad_model.return_value = SimpleNamespace(hidden_states=torch.zeros(1, 10))
        bad_model.config.id2label = {0: "AI", 1: "Real"}
        result = run_inference(rgb_image, bad_model, mock_processor)
        assert result["status"] == "error"
        assert result["error"]["code"] == "INFERENCE_ERROR"

    def test_error_tiene_clave_code(self, mock_model, mock_processor):
        result = run_inference("tipo_invalido", mock_model, mock_processor)
        assert "code" in result["error"]
        assert isinstance(result["error"]["code"], str)

    def test_error_tiene_clave_message(self, mock_model, mock_processor):
        result = run_inference("tipo_invalido", mock_model, mock_processor)
        assert "message" in result["error"]
        assert isinstance(result["error"]["message"], str)

    def test_error_scores_es_dict_vacio(self, mock_model, mock_processor):
        result = run_inference(b"bytes_corruptos", mock_model, mock_processor)
        assert result["scores"] == {{}}

    def test_error_label_id_es_none(self, mock_model, mock_processor):
        result = run_inference(b"bytes_corruptos", mock_model, mock_processor)
        assert result["label_id"] is None

    def test_error_tiene_timing_con_zeros(self, mock_model, mock_processor):
        result = run_inference(b"bytes_corruptos", mock_model, mock_processor)
        timing = result["timing"]
        assert "timing" in result
        assert timing["inference_ms"] == 0.0

    def test_proceso_continua_tras_imagen_invalida(self, rgb_image, mock_model, mock_processor):
        result_malo = run_inference(b"corrupto", mock_model, mock_processor)
        result_bueno = run_inference(rgb_image, mock_model, mock_processor)
        assert result_malo["status"] == "error"
        assert result_bueno["status"] == "ok"


# ---------------------------------------------------------------------------
# Tests - Timing (issue #11, sin cambios)
# ---------------------------------------------------------------------------

class TestRunInferenceTiming:

    def test_tiene_clave_timing(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "timing" in result

    def test_timing_es_dict(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result["timing"], dict)

    def test_timing_tiene_preprocessing_ms(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "preprocessing_ms" in result["timing"]

    def test_timing_tiene_inference_ms(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "inference_ms" in result["timing"]

    def test_timing_tiene_total_ms(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "total_ms" in result["timing"]

    def test_timing_valores_son_floats(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        assert isinstance(timing["preprocessing_ms"], float)
        assert isinstance(timing["inference_ms"], float)
        assert isinstance(timing["total_ms"], float)

    def test_timing_valores_son_positivos(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        assert timing["preprocessing_ms"] >= 0.0
        assert timing["inference_ms"] >= 0.0
        assert timing["total_ms"] >= 0.0

    def test_total_ms_es_suma_de_partes(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        expected = round(timing["preprocessing_ms"] + timing["inference_ms"], 3)
        assert abs(timing["total_ms"] - expected) < 0.01

    def test_timing_tiene_3_decimales(self, rgb_image, mock_model, mock_processor):
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        for key, val in timing.items():
            assert round(val, 3) == val, f"{key} tiene mas de 3 decimales: {val}"