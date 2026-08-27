.PHONY: setup run offline benchmark forecast dashboard test clean

setup:
	uv venv .venv 2>/dev/null || python3 -m venv .venv
	uv pip install -q -r requirements.txt 2>/dev/null || .venv/bin/pip install -q -r requirements.txt

run:            ## reconcile one batch, full pipeline
	.venv/bin/python -m khata.cli run
offline:        ## reconcile with the adjudicator off -- zero tokens
	.venv/bin/python -m khata.cli run --no-llm
benchmark:      ## six independent seeds, deterministic tiers only
	.venv/bin/python -m khata.cli benchmark --no-llm --seeds 6
benchmark-full: ## four independent seeds, full pipeline (spends tokens)
	.venv/bin/python -m khata.cli benchmark --seeds 4
forecast:       ## expected payouts still to come
	.venv/bin/python -m khata.cli forecast
dashboard:      ## http://127.0.0.1:8787
	.venv/bin/python -m khata.cli serve
test:
	.venv/bin/python -m pytest tests/ -q
clean:          ## drop generated artefacts -- but never hand-entered resolutions
	rm -rf data/*.jsonl .pytest_cache **/__pycache__
	@find data -maxdepth 1 -name '*.json' ! -name 'resolutions.json' -delete 2>/dev/null || true
