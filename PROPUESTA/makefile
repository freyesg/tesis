LATEX=pdflatex
BIBTEX=bibtex

MAIN=main
REF=referencias
FILENAME=propuesta
DIR=./

LATEXFLAGS= -shell-escape -jobname=$(FILENAME) -file-line-error -output-directory=$(DIR)

all: $(MAIN).tex $(REF).bib
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(BIBTEX) $(FILENAME).aux
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.lof *.lot *.toc *.cut ./contenido/*.aux *.bbl *.blg *.bcf

epub:
	latexml --dest=$(MAIN).xml $(MAIN).tex
	latexmlpost -dest=$(MAIN).html $(MAIN).xml
	ebook-convert $(MAIN).html $(MAIN).epub --language en --no-default-epub-cover
