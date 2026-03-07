"""Unit tests for app/clientGrpc.py.

Uses mocks so no real gRPC server is required.
All tests follow the AAA (Arrange, Act, Assert) pattern.
"""
from unittest.mock import MagicMock, patch

import grpc
import pytest

from app.clientGrpc import GRPCClient, GRPCClientError, _grpc_error_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_response(status_ok: bool = True) -> MagicMock:
    """Return a mock ClassificationResponse-like object."""
    response = MagicMock()
    response.image_id = "img-001"
    # inference_pb2.OK == 1, inference_pb2.ERROR == 2
    response.status = 1 if status_ok else 2
    response.predicted_label = "ai" if status_ok else ""
    response.confidence = 0.95
    response.prob_ai = 0.95
    response.prob_human = 0.05
    response.metrics.preprocess_time_ms = 10
    response.metrics.inference_time_ms = 50
    response.metrics.total_time_ms = 60
    response.error_message = "" if status_ok else "Invalid image"
    return response


# ---------------------------------------------------------------------------
# TestGRPCClientConnection
# ---------------------------------------------------------------------------

class TestGRPCClientConnection:
    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_conexion_exitosa(self, mock_stub_cls, mock_ready, mock_channel):
        # Arrange
        mock_ready.return_value.result.return_value = True
        mock_stub_cls.return_value = MagicMock()

        # Act
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Assert
        assert client._stub is not None
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    def test_timeout_lanza_grpc_client_error(self, mock_ready, mock_channel):
        # Arrange
        mock_ready.return_value.result.side_effect = Exception("connection timed out")

        # Act / Assert
        with pytest.raises(GRPCClientError):
            GRPCClient(host="localhost", port=12345, timeout=0)

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_close_limpia_canal(self, mock_stub_cls, mock_ready, mock_channel):
        # Arrange
        mock_ready.return_value.result.return_value = True
        mock_stub_cls.return_value = MagicMock()
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act
        client.close()

        # Assert
        assert client._channel is None
        assert client._stub is None

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_parametros_env_defaults(self, mock_stub_cls, mock_ready, mock_channel, monkeypatch):
        # Arrange: clear env vars so defaults are used
        monkeypatch.delenv("GRPC_SERVER_HOST", raising=False)
        monkeypatch.delenv("GRPC_SERVER_PORT", raising=False)
        monkeypatch.delenv("GRPC_TIMEOUT", raising=False)
        mock_ready.return_value.result.return_value = True
        mock_stub_cls.return_value = MagicMock()

        # Act
        client = GRPCClient()

        # Assert
        assert client.host == "localhost"
        assert client.port == 50051
        assert client.timeout == 5
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_parametros_env_custom(self, mock_stub_cls, mock_ready, mock_channel, monkeypatch):
        # Arrange: set custom env vars
        monkeypatch.setenv("GRPC_SERVER_HOST", "myhost")
        monkeypatch.setenv("GRPC_SERVER_PORT", "9999")
        monkeypatch.setenv("GRPC_TIMEOUT", "10")
        mock_ready.return_value.result.return_value = True
        mock_stub_cls.return_value = MagicMock()

        # Act
        client = GRPCClient()

        # Assert
        assert client.host == "myhost"
        assert client.port == 9999
        assert client.timeout == 10
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_constructor_override_env(self, mock_stub_cls, mock_ready, mock_channel, monkeypatch):
        # Arrange: env has one value but constructor provides another
        monkeypatch.setenv("GRPC_SERVER_HOST", "envhost")
        monkeypatch.setenv("GRPC_SERVER_PORT", "1111")
        mock_ready.return_value.result.return_value = True
        mock_stub_cls.return_value = MagicMock()

        # Act
        client = GRPCClient(host="overridehost", port=2222)

        # Assert: constructor values win
        assert client.host == "overridehost"
        assert client.port == 2222
        client.close()


# ---------------------------------------------------------------------------
# TestGRPCClientClassifyImage
# ---------------------------------------------------------------------------

class TestGRPCClientClassifyImage:
    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_classify_image_exitoso(self, mock_stub_cls, mock_ready, mock_channel):
        # Arrange
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.return_value = _make_dummy_response(status_ok=True)
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act
        result = client.classify_image(b"fakebytes", filename="test.jpg")

        # Assert
        assert result["status"] == "ok"
        assert result["predicted_label"] == "ai"
        assert 0.0 <= result["prob_ai"] <= 1.0
        assert 0.0 <= result["prob_real"] <= 1.0
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_classify_image_respuesta_error_servidor(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange: server returns ERROR status
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.return_value = _make_dummy_response(status_ok=False)
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act
        result = client.classify_image(b"invalidbytes", filename="bad.jpg")

        # Assert
        assert result["status"] == "error"
        assert result["error_message"] is not None
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_classify_image_rpc_error_lanza_excepcion(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange: RPC call raises grpc.RpcError
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.side_effect = grpc.RpcError()
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act / Assert
        with pytest.raises(GRPCClientError):
            client.classify_image(b"fakebytes")
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_classify_image_sin_stub_lanza_excepcion(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange
        mock_ready.return_value.result.return_value = True
        mock_stub_cls.return_value = MagicMock()
        client = GRPCClient(host="localhost", port=50051, timeout=1)
        client._stub = None  # Simulate disconnected state

        # Act / Assert
        with pytest.raises(GRPCClientError):
            client.classify_image(b"fakebytes")

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_classify_image_devuelve_campos_esperados(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.return_value = _make_dummy_response(status_ok=True)
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act
        result = client.classify_image(
            b"fakebytes", filename="test.jpg", image_id="img-test"
        )

        # Assert: all schema keys present
        expected_keys = (
            "image_id",
            "status",
            "predicted_label",
            "confidence",
            "prob_ai",
            "prob_real",
            "preprocess_time_ms",
            "inference_time_ms",
            "error_message",
        )
        for key in expected_keys:
            assert key in result, f"Missing key in result: {key}"
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_classify_image_probabilidades_en_rango(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.return_value = _make_dummy_response(status_ok=True)
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act
        result = client.classify_image(b"fakebytes")

        # Assert
        assert 0.0 <= result["prob_ai"] <= 1.0
        assert 0.0 <= result["prob_real"] <= 1.0
        assert abs(result["prob_ai"] + result["prob_real"] - 1.0) < 1e-4
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_classify_image_genera_image_id_si_no_se_pasa(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.return_value = _make_dummy_response(status_ok=True)
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act: no image_id provided
        client.classify_image(b"fakebytes")
        call_args = stub_instance.ClassifyImage.call_args

        # Assert: an image_id was generated
        sent_request = call_args[0][0]
        assert sent_request.image_id != ""
        client.close()


# ---------------------------------------------------------------------------
# TestGRPCClientErrorMessages
# ---------------------------------------------------------------------------

def _make_rpc_error(code: grpc.StatusCode, details: str = "detail") -> grpc.RpcError:
    """Create a real grpc.RpcError subclass instance with a specific status code."""

    class _FakeRpcError(grpc.RpcError):
        def code(self):
            return code

        def details(self):
            return details

    return _FakeRpcError()


class TestGRPCClientErrorMessages:
    """Tests for _grpc_error_message and status-code-specific error handling."""

    def test_grpc_error_message_deadline_exceeded(self):
        # Arrange
        err = _make_rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED)
        # Act
        msg = _grpc_error_message(err)
        # Assert: message mentions timeout
        assert "timeout" in msg.lower() or "excedió" in msg.lower()

    def test_grpc_error_message_unavailable(self):
        # Arrange
        err = _make_rpc_error(grpc.StatusCode.UNAVAILABLE)
        # Act
        msg = _grpc_error_message(err)
        # Assert: message mentions server not available
        assert "disponible" in msg.lower() or "unavailable" in msg.lower()

    def test_grpc_error_message_invalid_argument(self):
        # Arrange
        err = _make_rpc_error(grpc.StatusCode.INVALID_ARGUMENT)
        # Act
        msg = _grpc_error_message(err)
        # Assert: message mentions invalid payload/image
        assert "inválido" in msg.lower() or "invalid" in msg.lower()

    def test_grpc_error_message_internal(self):
        # Arrange
        err = _make_rpc_error(grpc.StatusCode.INTERNAL)
        # Act
        msg = _grpc_error_message(err)
        # Assert: message mentions internal/server error
        assert "interno" in msg.lower() or "internal" in msg.lower()

    def test_grpc_error_message_unknown_code_includes_code_name(self):
        # Arrange: use a status code without a predefined message
        err = _make_rpc_error(grpc.StatusCode.NOT_FOUND, details="resource missing")
        # Act
        msg = _grpc_error_message(err)
        # Assert: falls back to generic message with code name
        assert "NOT_FOUND" in msg or "resource missing" in msg

    def test_grpc_error_message_returns_string(self):
        # Arrange
        err = _make_rpc_error(grpc.StatusCode.CANCELLED)
        # Act / Assert
        assert isinstance(_grpc_error_message(err), str)

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_classify_image_deadline_exceeded_mensaje_amigable(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange: RPC raises DEADLINE_EXCEEDED
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.side_effect = _make_rpc_error(
            grpc.StatusCode.DEADLINE_EXCEEDED
        )
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act / Assert
        with pytest.raises(GRPCClientError) as exc_info:
            client.classify_image(b"fakebytes")
        assert "timeout" in str(exc_info.value).lower() or "excedió" in str(exc_info.value).lower()
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_classify_image_unavailable_mensaje_amigable(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange: RPC raises UNAVAILABLE
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.side_effect = _make_rpc_error(
            grpc.StatusCode.UNAVAILABLE
        )
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act / Assert
        with pytest.raises(GRPCClientError) as exc_info:
            client.classify_image(b"fakebytes")
        assert "disponible" in str(exc_info.value).lower() or "unavailable" in str(exc_info.value).lower()
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_classify_image_invalid_argument_mensaje_amigable(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange: RPC raises INVALID_ARGUMENT
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.side_effect = _make_rpc_error(
            grpc.StatusCode.INVALID_ARGUMENT
        )
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act / Assert
        with pytest.raises(GRPCClientError) as exc_info:
            client.classify_image(b"badbytes")
        assert "inválido" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    def test_connect_future_timeout_lanza_grpc_client_error(
        self, mock_ready, mock_channel
    ):
        # Arrange: channel_ready_future raises FutureTimeoutError
        mock_ready.return_value.result.side_effect = grpc.FutureTimeoutError()

        # Act / Assert
        with pytest.raises(GRPCClientError) as exc_info:
            GRPCClient(host="localhost", port=12345, timeout=0)
        assert "timeout" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# TestGRPCClientClassifyImageSafe
# ---------------------------------------------------------------------------

class TestGRPCClientClassifyImageSafe:
    """Tests for the classify_image_safe convenience method."""

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_safe_retorna_dict_en_exito(self, mock_stub_cls, mock_ready, mock_channel):
        # Arrange
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.return_value = _make_dummy_response(status_ok=True)
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act
        result = client.classify_image_safe(b"fakebytes", filename="test.jpg")

        # Assert
        assert isinstance(result, dict)
        assert result["status"] == "ok"
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_safe_no_lanza_en_error_rpc(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange: RPC raises UNAVAILABLE
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.side_effect = _make_rpc_error(
            grpc.StatusCode.UNAVAILABLE
        )
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act: must NOT raise
        result = client.classify_image_safe(b"fakebytes")

        # Assert
        assert result["status"] == "error"
        assert result["error_message"] is not None
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_safe_error_tiene_campos_esperados(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.side_effect = grpc.RpcError()
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act
        result = client.classify_image_safe(b"fakebytes", image_id="img-safe")

        # Assert: all schema keys must be present
        expected_keys = (
            "image_id",
            "status",
            "predicted_label",
            "confidence",
            "prob_ai",
            "prob_real",
            "preprocess_time_ms",
            "inference_time_ms",
            "error_message",
        )
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_safe_imagen_id_preservado_en_error(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.side_effect = grpc.RpcError()
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act
        result = client.classify_image_safe(b"fakebytes", image_id="img-preserve-me")

        # Assert
        assert result["image_id"] == "img-preserve-me"
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_safe_lote_continua_tras_imagen_invalida(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange: first call raises, second call succeeds
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.side_effect = [
            _make_rpc_error(grpc.StatusCode.INVALID_ARGUMENT),
            _make_dummy_response(status_ok=True),
        ]
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act: process two images; batch must not be aborted
        result_bad = client.classify_image_safe(b"bad", filename="bad.jpg")
        result_ok = client.classify_image_safe(b"good", filename="good.jpg")

        # Assert
        assert result_bad["status"] == "error"
        assert result_ok["status"] == "ok"
        client.close()

    @patch("app.clientGrpc.grpc.insecure_channel")
    @patch("app.clientGrpc.grpc.channel_ready_future")
    @patch("app.clientGrpc.inference_pb2_grpc.AiVsRealClassifierStub")
    def test_safe_error_message_es_string(
        self, mock_stub_cls, mock_ready, mock_channel
    ):
        # Arrange
        mock_ready.return_value.result.return_value = True
        stub_instance = MagicMock()
        stub_instance.ClassifyImage.side_effect = _make_rpc_error(
            grpc.StatusCode.DEADLINE_EXCEEDED
        )
        mock_stub_cls.return_value = stub_instance
        client = GRPCClient(host="localhost", port=50051, timeout=1)

        # Act
        result = client.classify_image_safe(b"fakebytes")

        # Assert: error_message must be a non-empty string
        assert isinstance(result["error_message"], str)
        assert len(result["error_message"]) > 0
        client.close()
