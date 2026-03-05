"""Tests for the real gRPC inference server.

Tests verify that the server starts correctly, responds with valid results
using a mock model/processor (no HuggingFace download required), handles
invalid images gracefully, and never crashes.

All tests follow the AAA (Arrange, Act, Assert) pattern.
"""
import io
import time
import types
from unittest.mock import MagicMock

import grpc
import pytest
import torch
from PIL import Image

import inference_pb2
import inference_pb2_grpc
from service.inference_server import serve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_png_bytes() -> bytes:
    """Return bytes of a minimal valid 10x10 RGB PNG image."""
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(128, 64, 32)).save(buf, format="PNG")
    return buf.getvalue()

def _make_mock_model():
    """Return a deterministic mock model (logits = [[3.0, 0.5]])."""
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    logits = torch.tensor([[3.0, 0.5]])
    mock_output = types.SimpleNamespace(logits=logits)
    mock_model.return_value = mock_output
    mock_model.config.id2label = {0: "ai", 1: "human"}
    return mock_model

def _make_mock_processor():
    """Return a mock processor that returns a dict with pixel_values tensor."""
    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    return mock_processor

def _make_request(image_id="img-001", filename="photo.png", image_data=None):
    if image_data is None:
        image_data = _make_valid_png_bytes()
    return inference_pb2.ImageRequest(
        image_id=image_id,
        filename=filename,
        image_data=image_data,
    )

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grpc_server_real():
    """Start the gRPC server with injected mock model/processor on port 50053."""
    model = _make_mock_model()
    processor = _make_mock_processor()
    server = serve(host="localhost", port=50053, model=model, processor=processor)
    time.sleep(0.2)
    yield server
    server.stop(grace=0)

@pytest.fixture(scope="module")
def grpc_stub(grpc_server_real):
    """Return a stub connected to the test server."""
    channel = grpc.insecure_channel("localhost:50053")
    stub = inference_pb2_grpc.AiVsRealClassifierStub(channel)
    yield stub
    channel.close()

# ---------------------------------------------------------------------------
# TestGrpcServerStartup
# ---------------------------------------------------------------------------

class TestGrpcServerStartup:
    def test_server_arranca(self, grpc_server_real):
        # Arrange / Act: fixture already starts server
        # Assert
        assert grpc_server_real is not None

# ---------------------------------------------------------------------------
# TestClassifyImageExitoso
# ---------------------------------------------------------------------------

class TestClassifyImageExitoso:
    def test_retorna_respuesta_no_nula(self, grpc_stub):
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert response is not None

    def test_image_id_coincide(self, grpc_stub):
        request = _make_request(image_id="img-abc")
        response = grpc_stub.ClassifyImage(request)
        assert response.image_id == "img-abc"

    def test_status_es_ok(self, grpc_stub):
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert response.status == inference_pb2.OK

    def test_predicted_label_es_ai_o_human(self, grpc_stub):
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert response.predicted_label in ("ai", "human")

    def test_predicted_label_es_determinista(self, grpc_stub):
        """Same image bytes always produce the same label (mock is deterministic)."""
        image_data = _make_valid_png_bytes()
        labels = {grpc_stub.ClassifyImage(_make_request(image_data=image_data)).predicted_label for _ in range(3)}
        assert len(labels) == 1

    def test_confidence_en_rango_valido(self, grpc_stub):
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert 0.0 <= response.confidence <= 1.0

    def test_prob_ai_en_rango_valido(self, grpc_stub):
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert 0.0 <= response.prob_ai <= 1.0

    def test_prob_human_en_rango_valido(self, grpc_stub):
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert 0.0 <= response.prob_human <= 1.0

    def test_probabilidades_suman_uno(self, grpc_stub):
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert abs(response.prob_ai + response.prob_human - 1.0) < 1e-4

    def test_metrics_no_es_none(self, grpc_stub):
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert response.metrics is not None

    def test_metrics_total_es_suma_de_partes(self, grpc_stub):
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        m = response.metrics
        assert m.total_time_ms == m.preprocess_time_ms + m.inference_time_ms

    def test_error_message_vacio_en_exito(self, grpc_stub):
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert response.error_message == ""

    def test_multiples_requests(self, grpc_stub):
        requests = [_make_request(image_id=f"img-{i}") for i in range(5)]
        responses = [grpc_stub.ClassifyImage(req) for req in requests]
        assert len(responses) == 5
        for i, resp in enumerate(responses):
            assert resp.image_id == f"img-{i}"
            assert resp.status == inference_pb2.OK


# ---------------------------------------------------------------------------
# TestClassifyImageError
# ---------------------------------------------------------------------------

class TestClassifyImageError:
    def test_bytes_invalidos_retorna_status_error(self, grpc_stub):
        request = _make_request(image_data=b"not-an-image")
        response = grpc_stub.ClassifyImage(request)
        assert response.status == inference_pb2.ERROR

    def test_bytes_vacios_retorna_status_error(self, grpc_stub):
        request = _make_request(image_data=b"")
        response = grpc_stub.ClassifyImage(request)
        assert response.status == inference_pb2.ERROR

    def test_error_message_no_vacio_en_error(self, grpc_stub):
        request = _make_request(image_data=b"bad-data")
        response = grpc_stub.ClassifyImage(request)
        assert response.error_message != ""

    def test_predicted_label_vacio_en_error(self, grpc_stub):
        request = _make_request(image_data=b"bad-data")
        response = grpc_stub.ClassifyImage(request)
        assert response.predicted_label == ""

    def test_servidor_continua_tras_imagen_invalida(self, grpc_stub):
        """Server must keep running after handling an invalid image."""
        grpc_stub.ClassifyImage(_make_request(image_data=b"bad-data"))
        response = grpc_stub.ClassifyImage(_make_request())
        assert response.status == inference_pb2.OK
