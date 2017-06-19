LATEX=pdflatex
BIBTEX=bibtex
CONTENT=chapters/

MAIN=tesis-postgrado
REF=referencias
FILENAME=tesis

LATEXFLAGS = -shell-escape -file-line-error -jobname=$(FILENAME)

all: $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(BIBTEX) $(FILENAME).aux
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.lof *.lot *.toc
	rm -f $(CONTENT)*.aux $(CONTENT)*.bbl $(CONTENT)*.blg $(CONTENT)*.log $(CONTENT)*.out $(CONTENT)*.lof $(CONTENT)*.lot $(CONTENT)*.toc
