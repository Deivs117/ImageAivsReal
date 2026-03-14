"""Tests for the real gRPC inference server.

Tests verify that the server starts correctly, responds with valid
results using a mock model/processor (no HuggingFace download required),
handles invalid images gracefully, and never crashes.

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
    """Return a mock processor with pixel_values tensor output."""
    mock_processor = MagicMock()
    mock_processor.return_value = {
        "pixel_values": torch.zeros(1, 3, 224, 224)
    }
    return mock_processor


def _make_request(image_id="img-001", filename="photo.png", image_data=None):
    """Build an ImageRequest with the given fields."""
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
    """Start gRPC server with mock model/processor on port 50053."""
    model = _make_mock_model()
    processor = _make_mock_processor()
    server = serve(
        host="localhost", port=50053, model=model, processor=processor
    )
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
    """Tests that the gRPC server starts without errors."""

    def test_server_arranca(self, grpc_server_real):
        """Server fixture must return a non-None server object."""
        # Arrange / Act: fixture already starts server
        # Assert
        assert grpc_server_real is not None


# ---------------------------------------------------------------------------
# TestClassifyImageExitoso
# ---------------------------------------------------------------------------


class TestClassifyImageExitoso:
    """Tests for successful image classification responses."""

    def test_retorna_respuesta_no_nula(self, grpc_stub):
        """ClassifyImage must return a non-None response."""
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert response is not None

    def test_image_id_coincide(self, grpc_stub):
        """Response image_id must match the request image_id."""
        request = _make_request(image_id="img-abc")
        response = grpc_stub.ClassifyImage(request)
        assert response.image_id == "img-abc"

    def test_status_es_ok(self, grpc_stub):
        """Response status must be OK for a valid image."""
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert response.status == inference_pb2.OK

    def test_predicted_label_es_ai_o_human(self, grpc_stub):
        """Predicted label must be either 'ai' or 'human'."""
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert response.predicted_label in ("ai", "human")

    def test_predicted_label_es_determinista(self, grpc_stub):
        """Same image bytes always produce the same label.

        The mock model is deterministic, so repeated calls with identical
        image data must return the same predicted_label every time.
        """
        image_data = _make_valid_png_bytes()
        labels = {
            grpc_stub.ClassifyImage(
                _make_request(image_data=image_data)
            ).predicted_label
            for _ in range(3)
        }
        assert len(labels) == 1

    def test_confidence_en_rango_valido(self, grpc_stub):
        """Confidence score must be in the [0.0, 1.0] range."""
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert 0.0 <= response.confidence <= 1.0

    def test_prob_ai_en_rango_valido(self, grpc_stub):
        """prob_ai must be in the [0.0, 1.0] range."""
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert 0.0 <= response.prob_ai <= 1.0

    def test_prob_human_en_rango_valido(self, grpc_stub):
        """prob_human must be in the [0.0, 1.0] range."""
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert 0.0 <= response.prob_human <= 1.0

    def test_probabilidades_suman_uno(self, grpc_stub):
        """prob_ai and prob_human must sum to approximately 1.0."""
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert abs(response.prob_ai + response.prob_human - 1.0) < 1e-4

    def test_metrics_no_es_none(self, grpc_stub):
        """Response metrics must not be None."""
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert response.metrics is not None

    def test_metrics_total_es_suma_de_partes(self, grpc_stub):
        """total_time_ms must equal preprocess_time_ms + inference_time_ms."""
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        m = response.metrics
        assert (
            m.total_time_ms == m.preprocess_time_ms + m.inference_time_ms
        )

    def test_error_message_vacio_en_exito(self, grpc_stub):
        """error_message must be empty for a successful response."""
        request = _make_request()
        response = grpc_stub.ClassifyImage(request)
        assert response.error_message == ""

    def test_multiples_requests(self, grpc_stub):
        """Multiple consecutive requests must all return status OK."""
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
    """Tests for error responses when the image is invalid."""

    def test_bytes_invalidos_retorna_status_error(self, grpc_stub):
        """Invalid image bytes must return status ERROR."""
        request = _make_request(image_data=b"not-an-image")
        response = grpc_stub.ClassifyImage(request)
        assert response.status == inference_pb2.ERROR

    def test_bytes_vacios_retorna_status_error(self, grpc_stub):
        """Empty image bytes must return status ERROR."""
        request = _make_request(image_data=b"")
        response = grpc_stub.ClassifyImage(request)
        assert response.status == inference_pb2.ERROR

    def test_error_message_no_vacio_en_error(self, grpc_stub):
        """error_message must not be empty when status is ERROR."""
        request = _make_request(image_data=b"bad-data")
        response = grpc_stub.ClassifyImage(request)
        assert response.error_message != ""

    def test_predicted_label_vacio_en_error(self, grpc_stub):
        """predicted_label must be empty when status is ERROR."""
        request = _make_request(image_data=b"bad-data")
        response = grpc_stub.ClassifyImage(request)
        assert response.predicted_label == ""

    def test_servidor_continua_tras_imagen_invalida(self, grpc_stub):
        """Server must keep running after handling an invalid image."""
        grpc_stub.ClassifyImage(_make_request(image_data=b"bad-data"))
        response = grpc_stub.ClassifyImage(_make_request())
        assert response.status == inference_pb2.OK


# ---------------------------------------------------------------------------
# TestClassifyImageGrpcStatusCodes
# ---------------------------------------------------------------------------


class TestClassifyImageGrpcStatusCodes:
    """Verify gRPC-level error handling: server stays alive after failures."""

    def test_servidor_no_colapsa_tras_multiples_errores(self, grpc_stub):
        """Multiple consecutive bad images must not crash the server."""
        for _ in range(3):
            response = grpc_stub.ClassifyImage(
                _make_request(image_data=b"bad")
            )
            assert response.status == inference_pb2.ERROR
        # Server must still respond to a valid request
        response = grpc_stub.ClassifyImage(_make_request())
        assert response.status == inference_pb2.OK

    def test_imagen_invalida_retorna_error_message(self, grpc_stub):
        """Invalid image must include a non-empty error_message."""
        response = grpc_stub.ClassifyImage(
            _make_request(image_data=b"not-an-image")
        )
        assert response.error_message != ""

    def test_imagen_invalida_no_lanza_excepcion_grpc(self, grpc_stub):
        """Application-level errors must NOT raise a gRPC RpcError.

        Invalid images are handled at the application level
        (status=ERROR in the response body) so that batch processing
        can continue uninterrupted.
        """
        try:
            response = grpc_stub.ClassifyImage(
                _make_request(image_data=b"bad")
            )
            # If no exception: the response must have status=ERROR
            assert response.status == inference_pb2.ERROR
        except grpc.RpcError as exc:
            pytest.fail(
                f"Unexpected gRPC RpcError: {exc.code()} – "
                "errors should be encoded in the response body."
            )
