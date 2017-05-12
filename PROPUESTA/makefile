LATEX=pdflatex
BIBTEX=bibtex

MAIN=main
REF=referencias
FILENAME=informe_final
DIR=./

LATEXFLAGS= -shell-escape -jobname=$(FILENAME) -file-line-error -output-directory=$(DIR)

all: $(MAIN).tex $(REF).bib
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(BIBTEX) $(FILENAME).aux
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.lof *.lot *.toc *.cut ./contenido/*.aux *.bbl *.blg *.bcf
