PYTHON ?= python

.PHONY: install predict evaluate test

install:
	$(PYTHON) -m pip install -r requirements.txt

predict:
	$(PYTHON) run_submission.py \
		--train data/train.jsonl \
		--input data/validation_inputs.jsonl \
		--output outputs/validation_predictions.jsonl \
		--artifacts-dir outputs/artifacts

evaluate: predict
	$(PYTHON) evaluator/evaluate.py \
		--inputs data/validation_inputs.jsonl \
		--ground-truth data/validation_ground_truth.jsonl \
		--predictions outputs/validation_predictions.jsonl \
		--report outputs/validation_report.json

test:
	$(PYTHON) -m pytest -q
