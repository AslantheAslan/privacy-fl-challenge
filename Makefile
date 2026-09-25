PYTHON ?= python
IMAGE ?= privacy-fl-challenge

.PHONY: install predict evaluate test lint report stress docker-build docker-run

install:
	$(PYTHON) -m pip install -r requirements.txt

# Standard challenge command.
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
	$(PYTHON) -m pytest

lint:
	ruff check .

# Regenerates the committed analysis in reports/ (adds holdout metrics via --eval-labels).
report:
	$(PYTHON) run_submission.py \
		--train data/train.jsonl \
		--input data/validation_inputs.jsonl \
		--output reports/validation_run/validation_predictions.jsonl \
		--artifacts-dir reports/validation_run \
		--eval-labels data/validation_ground_truth.jsonl
	$(PYTHON) evaluator/evaluate.py \
		--inputs data/validation_inputs.jsonl \
		--ground-truth data/validation_ground_truth.jsonl \
		--predictions reports/validation_run/validation_predictions.jsonl \
		--report reports/validation_run/validation_report.json
	$(PYTHON) scripts/format_shift_eval.py --output reports/format_shift_deid.json

stress:
	$(PYTHON) scripts/format_shift_eval.py --verbose

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --network none -v "$(CURDIR)/outputs:/app/outputs" $(IMAGE)
