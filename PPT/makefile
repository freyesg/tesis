LATEX=pdflatex
BIBTEX=bibtex

MAIN=main
REF=referencias
FILENAME=ppt

LATEXFLAGS=-shell-escape -jobname=$(FILENAME)

all: $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	#$(BIBTEX) $(FILENAME).aux
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	make clean

clean:
	rm -r *.aux *.log *.out *.nav *.snm *.toc
