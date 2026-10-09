install:
	pip install -r requirements.txt

smoke-test:
	python src/train.py --smoke-test --device cpu

train:
	python src/train.py --epochs 25

eval:
	python src/evaluate.py
