"""Tests unitarios para service/inference/preprocessing.py

Todos los tests siguen el patron AAA (Arrange, Act, Assert).
Los tests son completamente unitarios: no requieren descarga del modelo real,
se usa un processor mock para evitar dependencias de red/GPU.

Comando Make asociado:
    make test-preprocessing  ->  Ejecuta solo los tests de este modulo
    make test                ->  Ejecuta todos los tests del proyecto
"""

import io
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from service.inference.preprocessing import preprocess_image


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    return processor

@pytest.fixture
def rgb_pil_image():
    return Image.new("RGB", (64, 64), color=(100, 150, 200))

@pytest.fixture
def rgba_pil_image():
    return Image.new("RGBA", (64, 64), color=(100, 150, 200, 128))

@pytest.fixture
def grayscale_pil_image():
    return Image.new("L", (64, 64), color=128)

@pytest.fixture
def valid_image_bytes(rgb_pil_image):
    buffer = io.BytesIO()
    rgb_pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestPreprocessImageFromPIL:

    def test_retorna_dict_con_pixel_values(self, rgb_pil_image, mock_processor):
        result = preprocess_image(rgb_pil_image, mock_processor)
        assert isinstance(result, dict)
        assert "pixel_values" in result

    def test_pixel_values_es_tensor(self, rgb_pil_image, mock_processor):
        result = preprocess_image(rgb_pil_image, mock_processor)
        assert isinstance(result["pixel_values"], torch.Tensor)

    def test_pixel_values_shape_correcta(self, rgb_pil_image, mock_processor):
        result = preprocess_image(rgb_pil_image, mock_processor)
        assert result["pixel_values"].shape == (1, 3, 224, 224)

    def test_processor_llamado_con_return_tensors_pt(self, rgb_pil_image, mock_processor):
        preprocess_image(rgb_pil_image, mock_processor)
        mock_processor.assert_called_once()
        assert mock_processor.call_args.kwargs.get("return_tensors") == "pt"

    def test_imagen_rgba_convertida_a_rgb(self, rgba_pil_image, mock_processor):
        assert rgba_pil_image.mode == "RGBA"
        preprocess_image(rgba_pil_image, mock_processor)
        imagen_pasada = mock_processor.call_args.kwargs.get("images")
        assert imagen_pasada.mode == "RGB"

    def test_imagen_grayscale_convertida_a_rgb(self, grayscale_pil_image, mock_processor):
        assert grayscale_pil_image.mode == "L"
        preprocess_image(grayscale_pil_image, mock_processor)
        imagen_pasada = mock_processor.call_args.kwargs.get("images")
        assert imagen_pasada.mode == "RGB"

    def test_imagen_ya_rgb_no_se_reconvierte(self, rgb_pil_image, mock_processor):
        assert rgb_pil_image.mode == "RGB"
        preprocess_image(rgb_pil_image, mock_processor)
        imagen_pasada = mock_processor.call_args.kwargs.get("images")
        assert imagen_pasada.mode == "RGB"


class TestPreprocessImageFromBytes:

    def test_acepta_bytes_png_validos(self, valid_image_bytes, mock_processor):
        result = preprocess_image(valid_image_bytes, mock_processor)
        assert "pixel_values" in result

    def test_bytes_invalidos_lanza_value_error(self, mock_processor):
        with pytest.raises(ValueError, match="No se pudo decodificar los bytes"):
            preprocess_image(b"esto_no_es_una_imagen_valida", mock_processor)

    def test_bytes_vacios_lanza_value_error(self, mock_processor):
        with pytest.raises(ValueError, match="No se pudo decodificar los bytes"):
            preprocess_image(b"", mock_processor)


class TestPreprocessImageErrores:

    def test_tipo_invalido_string_lanza_type_error(self, mock_processor):
        with pytest.raises(TypeError, match="Se esperaba PIL.Image.Image o bytes"):
            preprocess_image("ruta/imagen.jpg", mock_processor)

    def test_tipo_invalido_int_lanza_type_error(self, mock_processor):
        with pytest.raises(TypeError, match="Se esperaba PIL.Image.Image o bytes"):
            preprocess_image(12345, mock_processor)

    def test_tipo_invalido_none_lanza_type_error(self, mock_processor):
        with pytest.raises(TypeError, match="Se esperaba PIL.Image.Image o bytes"):
            preprocess_image(None, mock_processor)

    def test_processor_falla_lanza_runtime_error(self, rgb_pil_image):
        broken_processor = MagicMock(side_effect=Exception("processor crashed"))
        with pytest.raises(RuntimeError, match="Error al procesar la imagen"):
            preprocess_image(rgb_pil_image, broken_processor)

    def test_processor_sin_pixel_values_lanza_value_error(self, rgb_pil_image):
        bad_processor = MagicMock(return_value={"logits": torch.zeros(1, 2)})
        with pytest.raises(ValueError, match="El processor no retorno 'pixel_values'"):
            preprocess_image(rgb_pil_image, bad_processor)

    def test_processor_retorna_dict_vacio_lanza_value_error(self, rgb_pil_image):
        empty_processor = MagicMock(return_value={})
        with pytest.raises(ValueError, match="El processor no retorno 'pixel_values'"):
            preprocess_image(rgb_pil_image, empty_processor)