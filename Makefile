PYTHON ?= python
PIP ?= $(PYTHON) -m pip
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: venv install install-dev lock format lint test prefetch convert clean

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(ACTIVATE) && $(PIP) install --upgrade pip
	$(ACTIVATE) && $(PIP) install -r requirements.txt

install-dev:
	$(ACTIVATE) && $(PIP) install --upgrade pip
	$(ACTIVATE) && $(PIP) install -r requirements-dev.txt

lock:
	$(ACTIVATE) && $(PIP) install pip-tools
	$(ACTIVATE) && pip-compile --resolver=backtracking --output-file requirements.lock.txt requirements.txt

format:
	$(ACTIVATE) && $(PYTHON) -m black src tests

lint:
	$(ACTIVATE) && $(PYTHON) -m ruff check src tests

test:
	$(ACTIVATE) && $(PYTHON) -m pytest -q

prefetch:
	$(ACTIVATE) && $(PYTHON) -m docling_offline.cli prefetch-models

convert:
	$(ACTIVATE) && $(PYTHON) -m docling_offline.cli convert $(INPUTS) --output out --format md json

clean:
	rm -rf $(VENV) .artifacts out .pytest_cache .mypy_cache
