PYTHON := python3

SCRIPTS := ch4.py co.py cubane.py ethylene.py
TARGETS := $(SCRIPTS:.py=.done)

.PHONY: all clean

all: $(TARGETS)

%.done: %.py
	@echo "Running $<"
	@$(PYTHON) $<

clean:
	rm -f *.abo
