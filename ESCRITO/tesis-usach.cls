% ib-assignment.cls
% Copyright 2009 Ankur Dave <ankurdave@gmail.com>
%
% Licensed under the CC-GNU GPL version 2.0 or later.
% See http://creativecommons.org/licenses/GPL/2.0
%
% See http://ankurdave.com/tex for more document classes.
% https://github.com/gmartinezramirez/Master-Thesis/blob/master/tesis-usach.cls
\ProvidesClass{tesis-usach}

\newif\if@escrito
\DeclareOption{tesis}{\@escritotrue}

\newif\if@inf
\DeclareOption{informe}{\@inftrue}

\newif\if@desc
\DeclareOption{descrip}{\@desctrue}

\newif\ifpropuesta\propuestafalse
\DeclareOption{propuesta}{\propuestatrue}
\ProcessOptions\relax

\if@escrito
	\LoadClass[10pt, oneside]{book}
	\setcounter{secnumdepth}{3}
	\setcounter{tocdepth}{3}

	\newcommand{\bibliografia}{\addcontentsline{toc}{chapter}{Bibliografía}}
\fi

\if@inf
	\LoadClass[12pt, oneside]{article}
	\RequirePackage{titlesec}
	%\setcounter{secnumdepth}{3}
	%\setcounter{tocdepth}{3}

	\newcommand{\bibliografia}{\addcontentsline{toc}{section}{Bibliografía}}
\fi

\newcommand{\autor}{\@author}
\renewcommand{\rmdefault}{phv} % Arial
\renewcommand{\sfdefault}{phv} % Arial




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Personalización de los títulos de secciones
\RequirePackage{titlesec}
%\renewcommand{\thesection}{\thechapter.\arabic{section}}

\titleformat{\chapter}{\bfseries\Large\vspace*{-4.0cm}}{\filright\Large\MakeUppercase\chaptertitlename\ \thechapter.\ }{0pt}{\filright\MakeUppercase}[\vspace{1.5pt}]
\titleformat{\section}{\bfseries\large\vspace{2pt}}{\large\MakeUppercase\thesection\ \vspace{2pt} }{0pt}{\MakeUppercase}[\vspace*{0.5cm}]
\titleformat{\subsection}{\bfseries\normalsize\vspace{2pt}}{\normalsize\thesubsection\ }{0pt}{\vspace*{0.5cm}}
\titleformat{\subsubsection}{\itshape\normalsize\vspace{1.0cm}}{\itshape\thesubsubsection\ }{0pt}{\vspace*{0.5cm}\itshape}
%
\titlespacing*{\chapter} {0pt}{85pt}{20pt}
\titlespacing*{\section} {0pt}{6.5ex plus 1ex minus .2ex}{2.3ex plus .2ex}
\titlespacing*{\subsection} {0pt}{6.5ex plus 1ex minus .2ex}{2.3ex plus .2ex}
\titlespacing*{\subsubsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
\titlespacing*{\paragraph} {0pt}{3.25ex plus 1ex minus .2ex}{1em}
\titlespacing*{\subparagraph} {\parindent}{3.25ex plus 1ex minus .2ex}{1em}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tamaño de la hoja
\RequirePackage{anysize} % Soporte para el comando \marginsize
\RequirePackage[papersize={21.6cm, 27.9cm}, letterpaper]{geometry} % , showframe
\geometry{left=4cm, top=2.5cm, right=2.5cm, bottom=2.5cm}%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Idioma español y carácteres utf8x
%\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{url}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figuras y tablas
\RequirePackage[pdftex]{graphicx} % PDFLaTeX
%\RequirePackage[font=small]{caption}
\RequirePackage{tikz}
\RequirePackage{float}
\RequirePackage{multicol}
\RequirePackage{multirow}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{comment}
\RequirePackage{color} % http://latexcolor.com/
\definecolor{gray}{rgb}{0.5, 0.5, 0.5}


\RequirePackage[spanish, es-lcroman, es-tabla, es-noquoting]{babel}
\RequirePackage{enumerate}
\RequirePackage{appendix}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bibliografía y referencias
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[apaciteclassic]{apacite}
%\RequirePackage{tocbibind} % Agregar referencias al índice
\RequirePackage[nottoc, notlof, notlot]{tocbibind}%evita que índice de contenidos, ilustraciones y tablas aparezca en tabla de contenido.
%\setcounter{secnumdepth}{2} %Mostrar numeración en las subsecciones (subsubsecciones no llevan números)
%\setcounter{tocdepth}{2} %Mostrar las subsubsecciones en el índice

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fuente matemática
\RequirePackage{amssymb}
\RequirePackage{amsmath}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estilo de página
\RequirePackage{fancyhdr}
\renewcommand{\headrulewidth}{0pt}
\pagestyle{fancy}
\lfoot{}
\cfoot{}
\rfoot{\thepage}
\lhead{}
\chead{}
\rhead{}

\fancypagestyle{plain}{
	\fancyhead[L, C, R]{}
	\fancyfoot[L, C]{}
	\fancyfoot[R]{\thepage}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identación, interlineado y spacing
\RequirePackage{setspace}
\RequirePackage{indentfirst}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3mm}
\linespread{1.5} %\renewcommand{\baselinestretch}{1.5}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% references
\RequirePackage{varioref}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lorem ipsum
\RequirePackage{lipsum}
\makeatletter
	\renewcommand\lips@dolipsum{%
		\ifnum\value{lips@count}<\lips@max\relax
			\addtocounter{lips@count}{1}%
			\csname lipsum@\romannumeral\c@lips@count\endcsname
			\lips@dolipsum
		\fi
	}
\makeatother
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Para crear argumentos para los comandos
\RequirePackage{xkeyval}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









\newcommand{\signature}{\underline{\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,}}







%Definiciones
\newtheorem{defin}{Definición}
\newtheorem{teorema}{Teorema}
\newtheorem{obs}{Observación}
%\setlength{\parindent}{2cm} % Sangrado
%\renewcommand{\baselinestretch}{1.5}                                    % Espaciado 1.5
\renewcommand{\and}{\\&}

\def\namedlabel#1#2{\begingroup
    #2%
    \def\@currentlabel{#2}%
    \phantomsection\label{#1}\endgroup
}

% fields that user can set (in addition to title, author, etc)
\makeatletter
	\newcommand{\candidatenum}[1]{\def \@candidatenum {#1}}
	\newcommand{\subtitle}[1]{\def \@subtitle {#1}}
	\newcommand{\proposito}[1]{\def \@proposito {#1}}
	\newcommand{\subject}[1]{\def \@subject {#1}}
	\newcommand{\wordcount}[1]{\def \@wordcount {#1}}
	\newcommand{\email}[1]{\def \@email {#1}}

	%\newcommand{\depto}{Departamento de Matemática y Ciencia de la Computación}
	%\newcommand{\facultad}{Facultad de Ciencia}
	\newcommand{\usach}{Universidad de Santiago de Chile}

	\newcommand{\upper}[1]{\expandafter\MakeUppercase\expandafter{#1}}



	% Departamento
	\def\departamento#1{\gdef\@depto{#1}}
	\def\@depto{\@latex@warning@no@line{No \noexpand\departamento given}}

	% Facultad
	\def\facultad#1{\gdef\@facultad{#1}}
	\def\@facultad{\@latex@warning@no@line{No \noexpand\facultad given}}

	% Profesor Guía
	\def\profesor#1{\gdef\@prof{#1}}
	\def\@prof{\@latex@warning@no@line{No \noexpand\profesor given}}

	% Revisores
	\def\revisores#1#2{\gdef\@revisorU{#1} \gdef\@revisorD{#2}}
	\def\@revisores{\@latex@warning@no@line{No \noexpand\revisores given. Faltan Revisores}}

	% Director Departamento
	\def\director#1{\gdef\@director{#1}}
	\def\@director{\@latex@warning@no@line{No \noexpand\director given}}

	% Informe
	\def\informe#1{\gdef\@informe{#1}}
	\def\@informe{\@latex@warning@no@line{No \noexpand\informe given}}

	% Tema
	\def\tema#1{\gdef\@tema{#1}}
	\def\@tema{\@latex@warning@no@line{No \noexpand\tema given}}

	% Celular
	\def\celular#1{\gdef\@celular{#1}}
	\def\@celular{\@latex@warning@no@line{No \noexpand\celular given}}

	% Correo
	\def\correo#1{\gdef\@correo{#1}}
	\def\@correo{\@latex@warning@no@line{No \noexpand\correo given}}

	% Correo
	\def\programa#1{\gdef\@programa{#1}}
	\def\@programa{\@latex@warning@no@line{No \noexpand\programa given}}

	\informe{}
	\tema{}
	\programa{}
	\celular{}
	\correo{}
	\facultad{}
	\departamento{}
	\candidatenum{}
	\subtitle{}
	\subject{}
	\wordcount{}
	\email{}
	\profesor{}
	\director{}
	\revisores{}
\makeatother




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Declaración de Entornos
%
% Agradecimientos
\newenvironment{agradecimientos}{
  \chapter*{Agradecimientos} % si no queremos que añada la palabra "Capitulo"
  \addcontentsline{toc}{chapter}{Agradecimientos} % si queremos que aparezca en el índice
}{}
%
% Resumen
\newenvironment{resumen}{
  \chapter*{Resumen} % si no queremos que añada la palabra "Capitulo"
  \addcontentsline{toc}{chapter}{Resumen} % si queremos que aparezca en el índice
}{}
%
% Dedicatoria
\newenvironment{dedicatoria}{
  \chapter*{}
  \begin{verse}
  \begin{flushright}\emph
} {
  \end{flushright}
  \end{verse}
}
%
% Fin Declaración de Entornos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Índice
\newcommand{\indice}{
	\tableofcontents
	%\newpage
	\cleardoublepage
	\listoftables
	\cleardoublepage
	\listoffigures
	\newpage
}



\if@escrito
\makeatletter
\def \maketitle {%
	\newgeometry{left=4cm, top=4cm, right=2.5cm, bottom=2.5cm}
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% PORTADA
	\setlength{\unitlength}{1cm} %Especificar unidad de trabajo
	\thispagestyle{empty}
	\begin{figure}[h]
		\begin{minipage}{0.05\textwidth}
			\begin{center}
			\end{center}
		\end{minipage}
		\begin{minipage}{0.75\textwidth}
			\begin{center}
				\vspace{0.5cm}
				{\fontsize{14}{1} {\bf \upper{\usach}}}\\
				{\fontsize{12}{1} {\bf \upper{\@facultad}}}\\
				{\fontsize{12}{1} {\bf \upper{\@depto}}}
			\end{center}
		\end{minipage}
		\begin{minipage}{0.15\textwidth}
			\begin{center}
				%\includegraphics[scale=0.06]{img/usach.png}
				\scalebox{0.17}{\input{img/logo}}
			\end{center}
		\end{minipage}
	\end{figure}

	\vfill
	\begin{center}
		{\fontsize{12}{1} \upper{{\bf \@title}}}\\
		{\fontsize{10}{1} \upper{{\bf \@subtitle}}}
	\end{center}

 	\vspace{2cm}
	\begin{center}
        {\fontsize{10}{1}
		\begin{tabular}{cc}
			\@author\\
		\end{tabular}
        }
	\end{center}

	\vfill
    \begin{minipage}[h]{0.36\textwidth}
    \hspace*{0.0cm}
    \end{minipage}
    \begin{minipage}[h]{0.5\textwidth}
    \begin{flushleft}
        {\fontsize{10}{1} Profesor Guía: \@prof}\\
        {\fontsize{10}{1} Tesis de grado presentada en conformidad a los requisitos para obtener el grado de Magíster en Ingeniería Informática}
    \end{flushleft}
    \end{minipage}

	\vfill
	\begin{center}
		{\fontsize{10}{1} Santiago, Chile\\ \the\year}
	\end{center}
	\newpage
	\restoregeometry
}
\makeatother
\fi



\if@inf
\makeatletter
\def \maketitle {%
	\newgeometry{left=4cm, right=2.5cm, top=4cm, bottom=2.5cm}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% PORTADA
	\setlength{\unitlength}{1cm} %Especificar unidad de trabajo
	\thispagestyle{empty}
	\begin{center}
		\vspace{0.5cm}
		{\fontsize{14}{1} {\bf \upper{\usach}}}\\
		{\fontsize{12}{1} {\bf \upper{\@facultad}}}\\
		{\fontsize{12}{1} {\bf \upper{\@depto}}}
	\end{center}

	\vfill
	\begin{center}
		{\fontsize{12}{1} {\bf \@title}}\\
		{\fontsize{10}{1} \upper{{\bf \@subtitle}}}
	\end{center}

	\vspace{1em}
	\begin{center}
		{\fontsize{12}{1} {\bf \@informe}}
	\end{center}

	\vfill
	\begin{minipage}[h]{0.36\textwidth}
	\hspace*{0.0cm}
	\end{minipage}
	\begin{minipage}[h]{0.5\textwidth}
	\begin{flushleft}
	\begin{tabular}{r}
		{\fontsize{10}{1} Nombre: \@author}\\
		{\fontsize{10}{1} Programa: \@programa}\\
		{\fontsize{10}{1} Profesor patrocinante: \@prof}\\
		{\fontsize{10}{1} Cel.: \@celular}\\
		{\fontsize{10}{1} email: \@correo}\\
		{\fontsize{10}{1} \@date}\\
	\end{tabular}
    \end{flushleft}
    \end{minipage}

	\newpage
	\restoregeometry
}
\makeatother
\fi













\newenvironment{descripcion}{
\scriptsize
	\begin{quote}\color{gray}
} {
	\end{quote}
	\bigskip
}
\newenvironment{arreglar}{
	\begingroup\color{red}
} {
	\endgroup
}
\newenvironment{falta}{
	\begingroup\color{red}
	\newline\newline\sc {\large Falta}
	\begin{itemize}
} {
\end{itemize}
	\endgroup
}

\RequirePackage{xkeyval}
\makeatletter
\define@key{tikzimg}{scale}{\def\escala{#1}}
\makeatother

\makeatletter
\newcommand{\makecopyright}{
	%\setkeys{makecopyright}{#1}%
	\thispagestyle{empty}
	\null
	\vfill
    \ifpropuesta
        % no va copyright
    \else
        \begin{flushleft}
		\copyright\;
			\autor - \the\year\\
			%\ifby
			\scalebox{0.3}{\input{img/by}}
			~$\bullet$~~ Algunos derechos reservados. Esta obra está bajo una Licencia Creative Commons Atribución-Chile 3.0. Sus condiciones de uso pueden ser revisadas en: \url{http://creativecommons.org/licenses/by/3.0/cl/}.
			%\fi
        \end{flushleft}
    \fi
}
\makeatother


\newenvironment{imagen}{
\setkeys{tikzimg}{scale=1.0}
	\begin{figure}[H]
	\centering
} {
	\end{figure}
}

%\hypersetup{
%bookmarksopen, bookmarksnumbered, pdfdisplaydoctitle=true,
%pdftitle=\@title,
%pdfauthor=\@autor,
%pdftoolbar=true,        % show Acrobat’s toolbar?
%pdfmenubar=true,        % show Acrobat’s menu?
%pdfsubject=Tesis de grado,
%pdfkeywords=,
%pdfcreator=Universidad de Santiago de Chile,
%pdfproducer=Felipe Reyes Gonz\'{a}lez,
%colorlinks=true,
%linkcolor=black,
%citecolor=black,
%filecolor=black,
%urlcolor=black,
%linkbordercolor={1 0 0},
%citebordercolor={0 1 0}
%}

% Cambios al default de spanish
\deactivatetilden
\decimalpoint

\addto\captionsspanish{
    \def\bibname{REFERENCIAS BIBLIOGRÁFICAS}
    \def\contentsname{TABLA DE CONTENIDO}
    \def\listtablename{ÍNDICE DE TABLAS}
    \def\listfigurename{ÍNDICE DE ILUSTRACIONES}
    \def\listalgorithmname{ÍNDICE DE ALGORITMOS}
    \def\appendixname{APÉNDICE}
    \def\tablename{Tabla}
    \def\figurename{Figura}
    \def\lstlistingname{Código}
    \def\chaptername{Capítulo}
}

\renewcommand{\appendixname}{APÉNDICES}
\renewcommand{\appendixtocname}{APÉNDICES}
\renewcommand{\appendixpagename}{APÉNDICES}

%Se agrega la palabra 'Figura' al índice de figuras
\usepackage{tocloft}
\renewcommand{\cftfigfont}{Figura }
%Lo mismo para el índice de tablas
\renewcommand{\cfttabfont}{Tabla }

%Se cambia el tamaño del título de la lista de figuras, tablas e índice de contenido dejándolos igual que los títulos normales
\renewcommand\cftloftitlefont{\Large\bf}
\renewcommand\cftlottitlefont{\Large\bf}
\renewcommand\cfttoctitlefont{\Large\bf}
