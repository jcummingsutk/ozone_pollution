install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C src/utils/*.py &&\
		pylint --disable=R,C src/models/*.py &&\
		pylint --disable=R,C *.py

all: install lint
