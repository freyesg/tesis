LATEX=pdflatex
BIBTEX=bibtex

MAIN=main
REF=referencias
FILENAME=tesis

LATEXFLAGS=-shell-escape -jobname=$(FILENAME) -file-line-error

all: $(MAIN).tex $(REF).bib
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(BIBTEX) $(FILENAME).aux
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	make clean

upgrade:
	git pull origin master




paohfpdofj
git:
	git add .
	git commit --author="freyesg <felipe.reyesg@usach.cl>" -m "ACTUALIZACIÓN $(shell date +%FT%T%Z)"
	git push origin master

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.lof *.lot *.toc *.cut ./contenido/*.aux
