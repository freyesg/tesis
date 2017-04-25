PYTHON=python
MAIN=main

all: $(MAIN).py
	@echo $(shell date +%FT%T%Z)
	$(PYTHON) $(MAIN).py
	make clean

clean:
	rm -f *.pyc *.png
