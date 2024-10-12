export PYTHONPATH = ./

run_train: 
	@echo "Running the model training..."
	python src/train.py

run_predict: 
	@echo "Running the model predict..."
	python src/predict.py 

all: run_train run_predict