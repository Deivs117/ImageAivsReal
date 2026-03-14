"""Unit tests for service/inference/preprocessing.py.

All tests follow the AAA (Arrange, Act, Assert) pattern.
Tests are fully unit-level: no real model download is required;
a mock processor avoids network/GPU dependencies.

Make targets:
    make test-preprocessing  ->  Run only tests for this module.
    make test                ->  Run all project tests.
"""
import io
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from service.inference.preprocessing import preprocess_image


@pytest.fixture
def mock_processor():
    """Return a mock processor with pixel_values (1, 3, 224, 224)."""
    processor = MagicMock()
    processor.return_value = {
        "pixel_values": torch.zeros(1, 3, 224, 224)
    }
    return processor


@pytest.fixture
def rgb_pil_image():
    """Return a 64x64 RGB PIL image."""
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


@pytest.fixture
def rgba_pil_image():
    """Return a 64x64 RGBA PIL image."""
    return Image.new("RGBA", (64, 64), color=(100, 150, 200, 128))


@pytest.fixture
def grayscale_pil_image():
    """Return a 64x64 grayscale (L-mode) PIL image."""
    return Image.new("L", (64, 64), color=128)


@pytest.fixture
def valid_image_bytes(rgb_pil_image):
    """Return a valid RGB image serialised as PNG bytes."""
    buffer = io.BytesIO()
    rgb_pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestPreprocessImageFromPIL:
    """Tests for preprocess_image when given a PIL image as input."""

    def test_retorna_dict_con_pixel_values(
        self, rgb_pil_image, mock_processor
    ):
        """Result must be a dict containing 'pixel_values'."""
        result = preprocess_image(rgb_pil_image, mock_processor)
        assert isinstance(result, dict)
        assert "pixel_values" in result

    def test_pixel_values_es_tensor(
        self, rgb_pil_image, mock_processor
    ):
        """pixel_values must be a torch.Tensor."""
        result = preprocess_image(rgb_pil_image, mock_processor)
        assert isinstance(result["pixel_values"], torch.Tensor)

    def test_pixel_values_shape_correcta(
        self, rgb_pil_image, mock_processor
    ):
        """pixel_values shape must be (1, 3, 224, 224)."""
        result = preprocess_image(rgb_pil_image, mock_processor)
        assert result["pixel_values"].shape == (1, 3, 224, 224)

    def test_processor_llamado_con_return_tensors_pt(
        self, rgb_pil_image, mock_processor
    ):
        """Processor must be called with return_tensors='pt'."""
        preprocess_image(rgb_pil_image, mock_processor)
        mock_processor.assert_called_once()
        assert (
            mock_processor.call_args.kwargs.get("return_tensors") == "pt"
        )

    def test_imagen_rgba_convertida_a_rgb(
        self, rgba_pil_image, mock_processor
    ):
        """RGBA image must be converted to RGB before processing."""
        assert rgba_pil_image.mode == "RGBA"
        preprocess_image(rgba_pil_image, mock_processor)
        imagen_pasada = mock_processor.call_args.kwargs.get("images")
        assert imagen_pasada.mode == "RGB"

    def test_imagen_grayscale_convertida_a_rgb(
        self, grayscale_pil_image, mock_processor
    ):
        """Grayscale image must be converted to RGB before processing."""
        assert grayscale_pil_image.mode == "L"
        preprocess_image(grayscale_pil_image, mock_processor)
        imagen_pasada = mock_processor.call_args.kwargs.get("images")
        assert imagen_pasada.mode == "RGB"

    def test_imagen_ya_rgb_no_se_reconvierte(
        self, rgb_pil_image, mock_processor
    ):
        """RGB image must be passed to processor unchanged."""
        assert rgb_pil_image.mode == "RGB"
        preprocess_image(rgb_pil_image, mock_processor)
        imagen_pasada = mock_processor.call_args.kwargs.get("images")
        assert imagen_pasada.mode == "RGB"


class TestPreprocessImageFromBytes:
    """Tests for preprocess_image when given bytes as input."""

    def test_acepta_bytes_png_validos(
        self, valid_image_bytes, mock_processor
    ):
        """Valid PNG bytes must produce a result with 'pixel_values'."""
        result = preprocess_image(valid_image_bytes, mock_processor)
        assert "pixel_values" in result

    def test_bytes_invalidos_lanza_value_error(self, mock_processor):
        """Invalid bytes must raise ValueError with descriptive message."""
        with pytest.raises(
            ValueError, match="No se pudo decodificar los bytes"
        ):
            preprocess_image(
                b"esto_no_es_una_imagen_valida", mock_processor
            )

    def test_bytes_vacios_lanza_value_error(self, mock_processor):
        """Empty bytes must raise ValueError with descriptive message."""
        with pytest.raises(
            ValueError, match="No se pudo decodificar los bytes"
        ):
            preprocess_image(b"", mock_processor)


class TestPreprocessImageErrores:
    """Tests for type and runtime error handling in preprocess_image."""

    def test_tipo_invalido_string_lanza_type_error(
        self, mock_processor
    ):
        """A string path must raise TypeError with descriptive message."""
        with pytest.raises(
            TypeError, match="Se esperaba PIL.Image.Image o bytes"
        ):
            preprocess_image("ruta/imagen.jpg", mock_processor)

    def test_tipo_invalido_int_lanza_type_error(self, mock_processor):
        """An integer must raise TypeError with descriptive message."""
        with pytest.raises(
            TypeError, match="Se esperaba PIL.Image.Image o bytes"
        ):
            preprocess_image(12345, mock_processor)

    def test_tipo_invalido_none_lanza_type_error(self, mock_processor):
        """None must raise TypeError with descriptive message."""
        with pytest.raises(
            TypeError, match="Se esperaba PIL.Image.Image o bytes"
        ):
            preprocess_image(None, mock_processor)

    def test_processor_falla_lanza_runtime_error(self, rgb_pil_image):
        """A crashing processor must raise RuntimeError."""
        broken_processor = MagicMock(
            side_effect=Exception("processor crashed")
        )
        with pytest.raises(
            RuntimeError, match="Error al procesar la imagen"
        ):
            preprocess_image(rgb_pil_image, broken_processor)

    def test_processor_sin_pixel_values_lanza_value_error(
        self, rgb_pil_image
    ):
        """Processor missing pixel_values must raise ValueError."""
        bad_processor = MagicMock(
            return_value={"logits": torch.zeros(1, 2)}
        )
        with pytest.raises(
            ValueError,
            match="El processor no retorno 'pixel_values'"
        ):
            preprocess_image(rgb_pil_image, bad_processor)

    def test_processor_retorna_dict_vacio_lanza_value_error(
        self, rgb_pil_image
    ):
        """Processor returning empty dict must raise ValueError."""
        empty_processor = MagicMock(return_value={})
        with pytest.raises(
            ValueError,
            match="El processor no retorno 'pixel_values'"
        ):
            preprocess_image(rgb_pil_image, empty_processor)
