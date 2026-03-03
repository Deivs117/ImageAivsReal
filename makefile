gui:
	@uv run -m streamlit run app/app.py

create_dirs:
	@echo "Creating directories..."
	mkdir service, proto, docs, data
	@echo "Directories created successfully."

grpc:
	@echo "gRPC stubs generating..."
	@if not exist proto\generated mkdir proto\generated
	@uv run python -m grpc_tools.protoc -I proto --python_out=proto/generated --grpc_python_out=proto/generated proto/inference.proto
	@echo "gRPC stubs generated in proto/generated/"

clean-grpc:
	@uv run python -c "import shutil; shutil.rmtree('proto/generated', ignore_errors=True)" && echo "gRPC stubs removed."