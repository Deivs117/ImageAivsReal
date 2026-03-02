create_dirs:
	@echo "Creating directories..."
	mkdir service, proto, docs, data
	@echo "Directories created successfully."

run_gui:
	python -m streamlit run app/app.py
	