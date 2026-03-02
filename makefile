ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif
run_gui:
	$(PY) -m streamlit run app/app.py
create_dirs:
	@echo "Creating directories..."
	mkdir service, proto, docs, data
	@echo "Directories created successfully."
grpc:
	@mkdir -p proto/generated
	$(PY) -m grpc_tools.protoc -I proto --python_out=proto/generated --grpc_python_out=proto/generated proto/inference.proto
	@echo "gRPC stubs generated in proto/generated/"
clean-grpc:
	@rm -rf proto/generated
	@echo "gRPC stubs removed."

