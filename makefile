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

