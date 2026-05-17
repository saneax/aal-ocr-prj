.PHONY: install run clean-leaks

install:
	bash scripts/install_deps.sh

run:
	.venv/bin/python main.py $(PDF) $(OUT)

clean-leaks:
	rm -f =*
