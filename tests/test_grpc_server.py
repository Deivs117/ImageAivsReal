"""Tests for gRPC inference server with real inference using mocked model/processor.

Tests verify that the server integrates correctly with run_inference using mocked
HuggingFace model and processor to avoid network downloads and GPU requirements.
All tests follow the AAA (Arrange, Act, Assert) pattern.
"""
import io
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import grpc
import pytest
import torch
from PIL import Image

import inference_pb2
import inference_pb2_grpc
from service.inference_server import serve


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope='module')
def mock_model():
    """Mock AutoModelForImageClassification with logits and id2label config."""
    model = MagicMock()
    model.return_value = SimpleNamespace(logits=torch.tensor([[3.0, 0.5]]))
    model.config.id2label = {0: 'ai', 1: 'human'}
    model.eval = MagicMock(return_value=model)
    return model


@pytest.fixture(scope='module')
def mock_processor():
    """Mock AutoImageProcessor that returns pixel_values tensor."""
    processor = MagicMock()
    processor.return_value = {'pixel_values': torch.zeros(1, 3, 224, 224)}
    return processor


@pytest.fixture(scope='module')
def valid_image_bytes():
    """Valid PNG image bytes (RGB 64x64)."""
    img = Image.new('RGB', (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture(scope='module')
def grpc_server_real(mock_model, mock_processor):
    """Start the gRPC server on port 50053 with mocked model and processor."""
    with patch(
        'service.inference_server.AutoModelForImageClassification.from_pretrained',
        return_value=mock_model,
    ), patch(
        'service.inference_server.AutoImageProcessor.from_pretrained',
        return_value=mock_processor,
    ):
        server = serve(host='localhost', port=50053)
    time.sleep(0.2)
    yield server
    server.stop(grace=0)


@pytest.fixture(scope='module')
def grpc_stub_real(grpc_server_real):
    """Return a stub connected to the mocked test server."""
    channel = grpc.insecure_channel('localhost:50053')
    stub = inference_pb2_grpc.AiVsRealClassifierStub(channel)
    yield stub
    channel.close()


def _make_valid_request(valid_image_bytes, image_id='img-001', filename='photo.png'):
    return inference_pb2.ImageRequest(
        image_id=image_id,
        filename=filename,
        image_data=valid_image_bytes,
    )


# ============================================================
# TestGrpcServerStartup
# ============================================================

class TestGrpcServerStartup:

    def test_server_startup(self, grpc_server_real):
        # Arrange / Act: fixture already starts server

        # Assert
        assert grpc_server_real is not None


# ============================================================
# TestClassifyImageExitoso
# ============================================================

class TestClassifyImageExitoso:

    def test_classify_image_basic(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes)

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert response is not None
        assert response.image_id == 'img-001'

    def test_response_status_ok(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes, image_id='img-ok')

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert response.status == inference_pb2.OK

    def test_response_predicted_label_es_string(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes, image_id='img-str')

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert isinstance(response.predicted_label, str)
        assert response.predicted_label != ''

    def test_response_predicted_label_valido(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes, image_id='img-label')

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert response.predicted_label in ('ai', 'human')

    def test_response_confidence_es_float(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes, image_id='img-conf')

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert isinstance(response.confidence, float)
        assert 0.0 <= response.confidence <= 1.0

    def test_response_prob_ai_rango_valido(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes, image_id='img-prob-ai')

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert 0.0 <= response.prob_ai <= 1.0

    def test_response_prob_human_rango_valido(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes, image_id='img-prob-human')

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert 0.0 <= response.prob_human <= 1.0

    def test_response_probs_suman_uno(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes, image_id='img-probs-sum')

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert abs(response.prob_ai + response.prob_human - 1.0) < 1e-4

    def test_response_error_message_vacio(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes, image_id='img-no-err')

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert response.error_message == ''

    def test_response_metrics_no_none(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes, image_id='img-metrics')

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert response.metrics is not None

    def test_response_metrics_total_correcto(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        request = _make_valid_request(valid_image_bytes, image_id='img-total')

        # Act
        response = grpc_stub_real.ClassifyImage(request)
        m = response.metrics

        # Assert
        assert m.total_time_ms == m.preprocess_time_ms + m.inference_time_ms

    def test_multiple_requests(self, grpc_stub_real, valid_image_bytes):
        # Arrange
        requests = [
            _make_valid_request(valid_image_bytes, image_id=f'img-multi-{i}')
            for i in range(5)
        ]

        # Act
        responses = [grpc_stub_real.ClassifyImage(req) for req in requests]

        # Assert
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.image_id == f'img-multi-{i}'
            assert response.status == inference_pb2.OK


# ============================================================
# TestClassifyImageError
# ============================================================

class TestClassifyImageError:

    def test_imagen_invalida_retorna_error(self, grpc_stub_real):
        # Arrange
        request = inference_pb2.ImageRequest(
            image_id='img-err-001',
            filename='bad.jpg',
            image_data=b'datos_invalidos',
        )

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert response.status == inference_pb2.ERROR

    def test_imagen_invalida_error_message_no_vacio(self, grpc_stub_real):
        # Arrange
        request = inference_pb2.ImageRequest(
            image_id='img-err-002',
            filename='bad.jpg',
            image_data=b'datos_invalidos',
        )

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert response.error_message != ''

    def test_imagen_invalida_predicted_label_vacio(self, grpc_stub_real):
        # Arrange
        request = inference_pb2.ImageRequest(
            image_id='img-err-003',
            filename='bad.jpg',
            image_data=b'datos_invalidos',
        )

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert response.predicted_label == ''

    def test_servidor_no_crashea_tras_error(self, grpc_stub_real, valid_image_bytes):
        # Arrange: first send invalid image
        bad_request = inference_pb2.ImageRequest(
            image_id='img-err-004',
            filename='bad.jpg',
            image_data=b'datos_invalidos',
        )
        good_request = _make_valid_request(valid_image_bytes, image_id='img-after-err')

        # Act
        bad_response = grpc_stub_real.ClassifyImage(bad_request)
        good_response = grpc_stub_real.ClassifyImage(good_request)

        # Assert
        assert bad_response.status == inference_pb2.ERROR
        assert good_response.status == inference_pb2.OK

    def test_bytes_vacios_retorna_error(self, grpc_stub_real):
        # Arrange
        request = inference_pb2.ImageRequest(
            image_id='img-err-005',
            filename='empty.jpg',
            image_data=b'',
        )

        # Act
        response = grpc_stub_real.ClassifyImage(request)

        # Assert
        assert response.status == inference_pb2.ERROR
