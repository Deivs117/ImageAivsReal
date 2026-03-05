"""Tests para service/inference_server.py con servidor real e inyección de mock.

El servidor usa run_inference() real pero con model/processor mockeados
para evitar descargas de HuggingFace y dependencias de GPU/red.

El mock está configurado con logits=[3.0, 0.5] e id2label={0:'ai', 1:'human'},
por lo que siempre predice 'ai' de forma determinista.

Fixtures:
    grpc_server: servidor iniciado en puerto 50052 con mocks inyectados
    grpc_stub: cliente conectado al servidor de test

Comandos Make:
    make test-grpc   -> Ejecuta solo los tests del servidor gRPC
    make test        -> Ejecuta todos los tests del proyecto
"""
import io
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import grpc
import pytest
import torch
from PIL import Image

import inference_pb2
import inference_pb2_grpc
from service.inference_server import serve


# ============================================================
# Helpers
# ============================================================

def _make_valid_png_bytes():
    """Genera bytes de una imagen PNG válida en memoria usando PIL."""
    img = Image.new('RGB', (32, 32), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _make_request(image_id='img-001', filename='photo.jpg', use_valid_image=True):
    """Crea un ImageRequest con imagen válida o bytes inválidos según use_valid_image."""
    if use_valid_image:
        image_data = _make_valid_png_bytes()
    else:
        image_data = b'\xff\xd8\xff'
    return inference_pb2.ImageRequest(
        image_id=image_id,
        filename=filename,
        image_data=image_data,
    )


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope='module')
def mock_model():
    """Mock determinista que simula un modelo HuggingFace de clasificación."""
    model = MagicMock()
    model.config.id2label = {0: 'ai', 1: 'human'}
    model.eval.return_value = None
    model.return_value = SimpleNamespace(
        logits=torch.tensor([[3.0, 0.5]])
    )
    return model


@pytest.fixture(scope='module')
def mock_processor():
    """Mock determinista que simula un AutoImageProcessor de HuggingFace."""
    processor = MagicMock()
    processor.return_value = {'pixel_values': torch.zeros(1, 3, 224, 224)}
    return processor


@pytest.fixture(scope='module')
def grpc_server(mock_model, mock_processor):
    """Inicia el servidor gRPC en el puerto 50052 con mocks inyectados."""
    server = serve(host='localhost', port=50052, model=mock_model, processor=mock_processor)
    time.sleep(0.2)
    yield server
    server.stop(grace=0)


@pytest.fixture(scope='module')
def grpc_stub(grpc_server):
    """Retorna un stub conectado al servidor de test."""
    channel = grpc.insecure_channel('localhost:50052')
    stub = inference_pb2_grpc.AiVsRealClassifierStub(channel)
    yield stub
    channel.close()


# ============================================================
# Tests
# ============================================================

class TestGrpcServerStartup:
    def test_server_not_none(self, grpc_server):
        # Arrange / Act: fixture ya inicia el servidor

        # Assert
        assert grpc_server is not None


class TestClassifyImageExitoso:
    """Tests con imágenes válidas; el mock siempre devuelve status OK y label 'ai'."""

    def test_retorna_response(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert response is not None

    def test_image_id_correcto(self, grpc_stub):
        # Arrange
        request = _make_request(image_id='img-001')

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert response.image_id == 'img-001'

    def test_status_es_ok(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert response.status == inference_pb2.OK

    def test_predicted_label_es_string(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert isinstance(response.predicted_label, str)

    def test_predicted_label_es_ai_o_human(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert response.predicted_label in ('ai', 'human')

    def test_predicted_label_es_ai_con_mock(self, grpc_stub):
        # Arrange: mock con logits [3.0, 0.5] → clase 0 = 'ai'
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert response.predicted_label == 'ai'

    def test_confidence_en_rango(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert 0.0 <= response.confidence <= 1.0

    def test_prob_ai_en_rango(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert 0.0 <= response.prob_ai <= 1.0

    def test_prob_human_en_rango(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert 0.0 <= response.prob_human <= 1.0

    def test_probabilidades_suman_uno(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert abs(response.prob_ai + response.prob_human - 1.0) < 1e-4

    def test_metrics_no_es_none(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert response.metrics is not None

    def test_metrics_total_es_suma(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)
        m = response.metrics

        # Assert
        assert m.total_time_ms == m.preprocess_time_ms + m.inference_time_ms

    def test_error_message_vacio(self, grpc_stub):
        # Arrange
        request = _make_request()

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert response.error_message == ''

    def test_multiples_requests_status_ok(self, grpc_stub):
        # Arrange
        requests = [_make_request(image_id=f'img-{i}') for i in range(5)]

        # Act
        responses = [grpc_stub.ClassifyImage(req) for req in requests]

        # Assert
        for response in responses:
            assert response.status == inference_pb2.OK


class TestClassifyImageError:
    """Tests con bytes inválidos; el servidor debe retornar status ERROR."""

    def test_bytes_invalidos_retorna_error(self, grpc_stub):
        # Arrange
        request = _make_request(use_valid_image=False)

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert response.status == inference_pb2.ERROR

    def test_bytes_invalidos_error_message_no_vacio(self, grpc_stub):
        # Arrange
        request = _make_request(use_valid_image=False)

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert len(response.error_message) > 0

    def test_bytes_invalidos_predicted_label_vacio(self, grpc_stub):
        # Arrange
        request = _make_request(use_valid_image=False)

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert response.predicted_label == ''

    def test_bytes_vacios_retorna_error(self, grpc_stub):
        # Arrange
        request = inference_pb2.ImageRequest(
            image_id='img-empty',
            filename='empty.jpg',
            image_data=b'',
        )

        # Act
        response = grpc_stub.ClassifyImage(request)

        # Assert
        assert response.status == inference_pb2.ERROR

    def test_servidor_continua_tras_error(self, grpc_stub):
        # Arrange: primera petición inválida
        invalid_request = _make_request(use_valid_image=False)
        valid_request = _make_request(image_id='img-recovery', use_valid_image=True)

        # Act
        grpc_stub.ClassifyImage(invalid_request)
        response = grpc_stub.ClassifyImage(valid_request)

        # Assert: el servidor sigue funcionando después del error
        assert response.status == inference_pb2.OK
