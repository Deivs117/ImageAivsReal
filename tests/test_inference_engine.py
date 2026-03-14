"""Unit tests for service/inference/inference_engine.py.

All tests follow the AAA (Arrange, Act, Assert) pattern.
Tests are fully unit-level: no real model download is required;
mock model and processor avoid network/GPU dependencies.

Make targets:
    make test-inference  ->  Run only tests for this module.
    make test            ->  Run all project tests.
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
    """Return a 64x64 RGB PIL image."""
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


@pytest.fixture
def valid_image_bytes(rgb_image):
    """Return a valid RGB image serialised as PNG bytes."""
    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_processor():
    """Return a mock processor with pixel_values (1, 3, 224, 224)."""
    processor = MagicMock()
    processor.return_value = {
        "pixel_values": torch.zeros(1, 3, 224, 224)
    }
    return processor


@pytest.fixture
def mock_model():
    """Return a mock model with logits (1,2) and id2label mapping."""
    model = MagicMock()
    logits = torch.tensor([[2.0, 0.5]])  # AI wins
    model.return_value = SimpleNamespace(logits=logits)
    model.config.id2label = {0: "AI", 1: "Real"}
    return model


# ---------------------------------------------------------------------------
# Tests - Successful output structure (updated - issue #12)
# ---------------------------------------------------------------------------


class TestRunInferenceOutput:
    """Tests that verify the structure of a successful run_inference result."""

    def test_retorna_dict(self, rgb_image, mock_model, mock_processor):
        """run_inference must return a dict."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result, dict)

    def test_tiene_clave_status(
        self, rgb_image, mock_model, mock_processor
    ):
        """Result dict must contain 'status' key."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "status" in result

    def test_status_ok_en_caso_exitoso(
        self, rgb_image, mock_model, mock_processor
    ):
        """status must be 'ok' for a successful inference."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert result["status"] == "ok"

    def test_tiene_clave_label(
        self, rgb_image, mock_model, mock_processor
    ):
        """Result dict must contain 'label' key."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "label" in result

    def test_tiene_clave_label_id(
        self, rgb_image, mock_model, mock_processor
    ):
        """Result dict must contain 'label_id' key."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "label_id" in result

    def test_tiene_clave_scores(
        self, rgb_image, mock_model, mock_processor
    ):
        """Result dict must contain 'scores' key."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "scores" in result

    def test_tiene_clave_error(
        self, rgb_image, mock_model, mock_processor
    ):
        """Result dict must contain 'error' key."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "error" in result

    def test_error_es_none_en_caso_exitoso(
        self, rgb_image, mock_model, mock_processor
    ):
        """error must be None for a successful inference."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert result["error"] is None

    def test_scores_es_dict(
        self, rgb_image, mock_model, mock_processor
    ):
        """scores value must be a dict."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result["scores"], dict)

    def test_scores_suman_aproximadamente_uno(
        self, rgb_image, mock_model, mock_processor
    ):
        """All score values must sum to approximately 1.0."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 1e-4

    def test_scores_son_floats(
        self, rgb_image, mock_model, mock_processor
    ):
        """Each score value must be a float."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        for v in result["scores"].values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Tests - Output values
# ---------------------------------------------------------------------------


class TestRunInferenceLabel:
    """Tests that verify label and scores values in the output."""

    def test_label_es_string(
        self, rgb_image, mock_model, mock_processor
    ):
        """label must be a string."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result["label"], str)

    def test_label_id_es_int(
        self, rgb_image, mock_model, mock_processor
    ):
        """label_id must be an integer."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result["label_id"], int)

    def test_label_esta_en_scores(
        self, rgb_image, mock_model, mock_processor
    ):
        """label value must be a key in scores."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert result["label"] in result["scores"]

    def test_label_correcto_segun_logits(
        self, rgb_image, mock_model, mock_processor
    ):
        """label must match the highest logit class (AI at index 0)."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert result["label"] == "AI"
        assert result["label_id"] == 0

    def test_label_real_cuando_logit_mayor(
        self, rgb_image, mock_processor
    ):
        """label must be 'Real' when its logit is the highest."""
        model = MagicMock()
        model.return_value = SimpleNamespace(
            logits=torch.tensor([[0.1, 3.0]])
        )
        model.config.id2label = {0: "AI", 1: "Real"}
        result = run_inference(rgb_image, model, mock_processor)
        assert result["label"] == "Real"
        assert result["label_id"] == 1

    def test_scores_contiene_todas_las_clases(
        self, rgb_image, mock_model, mock_processor
    ):
        """scores must contain keys for all classes in id2label."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert set(result["scores"].keys()) == {"AI", "Real"}


# ---------------------------------------------------------------------------
# Tests - Bytes input
# ---------------------------------------------------------------------------


class TestRunInferenceDesdeBytes:
    """Tests that verify run_inference accepts raw image bytes."""

    def test_acepta_bytes_png_validos(
        self, valid_image_bytes, mock_model, mock_processor
    ):
        """Valid PNG bytes must produce a successful result."""
        result = run_inference(
            valid_image_bytes, mock_model, mock_processor
        )
        assert result["status"] == "ok"
        assert "label" in result

    def test_bytes_invalidos_retorna_error_controlado(
        self, mock_model, mock_processor
    ):
        """Invalid bytes must return an error dict (no exception)."""
        # Arrange / Act - issue #12: does NOT raise, returns error dict
        result = run_inference(
            b"datos_invalidos", mock_model, mock_processor
        )
        # Assert
        assert result["status"] == "error"
        assert result["error"]["code"] == "INVALID_IMAGE"
        assert result["label"] is None
        assert result["scores"] == dict()


# ---------------------------------------------------------------------------
# Tests - Controlled error handling (updated - issue #12)
# ---------------------------------------------------------------------------


class TestRunInferenceErroresControlados:
    """Tests that verify controlled error handling in run_inference."""

    def test_tipo_invalido_retorna_error_controlado(
        self, mock_model, mock_processor
    ):
        """An invalid input type must return an error dict."""
        # Arrange / Act - does NOT raise TypeError
        result = run_inference(
            "ruta/imagen.jpg", mock_model, mock_processor
        )
        # Assert
        assert result["status"] == "error"
        assert result["error"]["code"] == "INVALID_IMAGE"
        assert result["label"] is None

    def test_modelo_falla_retorna_error_controlado(
        self, rgb_image, mock_processor
    ):
        """A failing model must return an error dict."""
        # Arrange
        broken_model = MagicMock(
            side_effect=Exception("CUDA out of memory")
        )
        broken_model.config.id2label = {0: "AI", 1: "Real"}
        # Act - does NOT raise RuntimeError
        result = run_inference(rgb_image, broken_model, mock_processor)
        # Assert
        assert result["status"] == "error"
        assert result["error"]["code"] == "INFERENCE_ERROR"
        assert "CUDA out of memory" in result["error"]["message"]

    def test_modelo_sin_logits_retorna_error_controlado(
        self, rgb_image, mock_processor
    ):
        """A model without logits in output must return an error dict."""
        # Arrange
        bad_model = MagicMock()
        bad_model.return_value = SimpleNamespace(
            hidden_states=torch.zeros(1, 10)
        )
        bad_model.config.id2label = {0: "AI", 1: "Real"}
        # Act - does NOT raise ValueError
        result = run_inference(rgb_image, bad_model, mock_processor)
        # Assert
        assert result["status"] == "error"
        assert result["error"]["code"] == "INFERENCE_ERROR"

    def test_error_tiene_clave_code(self, mock_model, mock_processor):
        """Error dict must contain a 'code' string key."""
        # Arrange / Act
        result = run_inference(
            "tipo_invalido", mock_model, mock_processor
        )
        # Assert
        assert "code" in result["error"]
        assert isinstance(result["error"]["code"], str)

    def test_error_tiene_clave_message(
        self, mock_model, mock_processor
    ):
        """Error dict must contain a 'message' string key."""
        # Arrange / Act
        result = run_inference(
            "tipo_invalido", mock_model, mock_processor
        )
        # Assert
        assert "message" in result["error"]
        assert isinstance(result["error"]["message"], str)

    def test_error_scores_es_dict_vacio(
        self, mock_model, mock_processor
    ):
        """scores must be an empty dict on error."""
        # Arrange / Act
        result = run_inference(
            b"bytes_corruptos", mock_model, mock_processor
        )
        # Assert: scores is an empty dict
        assert result["scores"] == dict()

    def test_error_label_id_es_none(self, mock_model, mock_processor):
        """label_id must be None on error."""
        # Arrange / Act
        result = run_inference(
            b"bytes_corruptos", mock_model, mock_processor
        )
        # Assert
        assert result["label_id"] is None

    def test_error_tiene_timing_con_zeros(
        self, mock_model, mock_processor
    ):
        """timing must be present; inference_ms must be 0.0 on error."""
        # Arrange / Act
        result = run_inference(
            b"bytes_corruptos", mock_model, mock_processor
        )
        timing = result["timing"]
        # Assert: timing always present; inference_ms=0 if failed early
        assert "timing" in result
        assert timing["inference_ms"] == 0.0

    def test_proceso_continua_tras_imagen_invalida(
        self, rgb_image, mock_model, mock_processor
    ):
        """Batch processing must not stop after an invalid image."""
        # Arrange: simulate a batch with one invalid and one valid image
        result_malo = run_inference(
            b"corrupto", mock_model, mock_processor
        )
        result_bueno = run_inference(
            rgb_image, mock_model, mock_processor
        )
        # Assert: batch is not interrupted
        assert result_malo["status"] == "error"
        assert result_bueno["status"] == "ok"


# ---------------------------------------------------------------------------
# Tests - Timing (issue #11, unchanged)
# ---------------------------------------------------------------------------


class TestRunInferenceTiming:
    """Tests that verify timing data in the run_inference result."""

    def test_tiene_clave_timing(
        self, rgb_image, mock_model, mock_processor
    ):
        """Result must contain 'timing' key."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "timing" in result

    def test_timing_es_dict(
        self, rgb_image, mock_model, mock_processor
    ):
        """timing value must be a dict."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert isinstance(result["timing"], dict)

    def test_timing_tiene_preprocessing_ms(
        self, rgb_image, mock_model, mock_processor
    ):
        """timing must contain 'preprocessing_ms' key."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "preprocessing_ms" in result["timing"]

    def test_timing_tiene_inference_ms(
        self, rgb_image, mock_model, mock_processor
    ):
        """timing must contain 'inference_ms' key."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "inference_ms" in result["timing"]

    def test_timing_tiene_total_ms(
        self, rgb_image, mock_model, mock_processor
    ):
        """timing must contain 'total_ms' key."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        assert "total_ms" in result["timing"]

    def test_timing_valores_son_floats(
        self, rgb_image, mock_model, mock_processor
    ):
        """All timing values must be floats."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        assert isinstance(timing["preprocessing_ms"], float)
        assert isinstance(timing["inference_ms"], float)
        assert isinstance(timing["total_ms"], float)

    def test_timing_valores_son_positivos(
        self, rgb_image, mock_model, mock_processor
    ):
        """All timing values must be non-negative."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        assert timing["preprocessing_ms"] >= 0.0
        assert timing["inference_ms"] >= 0.0
        assert timing["total_ms"] >= 0.0

    def test_total_ms_es_suma_de_partes(
        self, rgb_image, mock_model, mock_processor
    ):
        """total_ms must equal preprocessing_ms + inference_ms."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        expected = round(
            timing["preprocessing_ms"] + timing["inference_ms"], 3
        )
        assert abs(timing["total_ms"] - expected) < 0.01

    def test_timing_tiene_3_decimales(
        self, rgb_image, mock_model, mock_processor
    ):
        """Each timing value must have at most 3 decimal places."""
        result = run_inference(rgb_image, mock_model, mock_processor)
        timing = result["timing"]
        for key, val in timing.items():
            assert round(val, 3) == val, (
                f"{key} has more than 3 decimal places: {val}"
            )
