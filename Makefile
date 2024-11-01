export PYTHONPATH = ./

venv:
	python -m venv venv 
	venv\Scripts\activate && pip install -r requirements.txt

venv_run: 
	@echo "Running the model training..."
	venv\Scripts\activate && python src/train.py
	@echo "Running the model predict..."
	venv\Scripts\activate && python src/predict.py 

venv_remove:
	rmdir /S /Q venv

docker-build-run:
	docker build -t mlops .
	docker run mlops

all-run: venv venv_run venv_remove

docker: docker-build-run
