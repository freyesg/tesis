LATEX=pdflatex
BIBTEX=bibtex

MAIN=main
REF=referencias
FILENAME=tesis

LATEXFLAGS=-shell-escape -jobname=$(FILENAME) -file-line-error

all: $(MAIN).tex $(REF).bib
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	make clean

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.lof *.lot *.toc *.cut ./contenido/*.aux
