% ib-assignment.cls
% Copyright 2009 Ankur Dave <ankurdave@gmail.com>
%
% Licensed under the CC-GNU GPL version 2.0 or later.
% See http://creativecommons.org/licenses/GPL/2.0
%
% See http://ankurdave.com/tex for more document classes.

\ProvidesClass{tesis-usach}


\newif\if@uno
\DeclareOption{ppt1}{\@unotrue}

\newif\if@dos
\DeclareOption{ppt2}{\@dostrue}

\newif\if@tres
\DeclareOption{ppt3}{\@trestrue}

\newif\if@escrito
\DeclareOption{tesis}{\@escritotrue}

\newif\if@inf
\DeclareOption{informe}{\@inftrue}

\newif\if@desc
\DeclareOption{descrip}{\@desctrue}

\ProcessOptions\relax

%\if@infuno
%	\newcommand{\informe}{Informe N$^\circ$ 1}
%\fi
%\if@infdos
%	\newcommand{\informe}{Informe N$^\circ$ 2}
%\fi
%\if@inftres
%	\newcommand{\informe}{Informe N$^\circ$ 3}
%\fi

\newcommand{\inegrotrz}{\input{img/inegrotrz}}


\if@escrito
	\LoadClass[12pt, oneside]{article}
	\usepackage{titlesec}
	\titleformat{\chapter}[display]{\normalfont\bfseries}{\chaptertitlename\ \thechapter}{10pt}{}
	\titlespacing*{\chapter}{0pt}{0pt}{10pt}
	\titleformat{\section}[hang]{\bf}{{\bf \thesection\enspace}}{10pt}{}
	\titleformat{\subsection}[hang]{\bf}{{\bf \thesubsection\enspace}}{10pt}{}
	\titleformat{\subsubsection}[hang]{\bf}{{\bf \thesubsubsection\enspace}}{10pt}{}
	\setcounter{secnumdepth}{3}
	\setcounter{tocdepth}{3}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Interlineado
	\renewcommand{\baselinestretch}{1.5}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Tamaño de la hoja
	\usepackage{anysize} % Soporte para el comando \marginsize
	\usepackage[papersize={21.6cm, 27.9cm}]{geometry}
	\geometry{left=4cm, right=2.5cm, top=2.5cm, bottom=2.5cm}

	\newcommand{\bibliografia}{\addcontentsline{toc}{chapter}{Bibliografía}}

\fi

\if@inf
	\LoadClass[12pt, oneside]{article}
	\usepackage{titlesec}
	\titleformat{\section}[hang]{\bf}{{\bf \thesection\enspace}}{14pt}{}
	\titleformat{\subsection}[hang]{\bf}{{\bf \thesubsection\enspace}}{14pt}{}
	\titleformat{\subsubsection}[hang]{\bf}{{\bf \thesubsubsection\enspace}}{14pt}{}
	\setcounter{secnumdepth}{3}
	\setcounter{tocdepth}{3}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Interlineado
	\renewcommand{\baselinestretch}{3}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Tamaño de la hoja
	\usepackage{anysize} % Soporte para el comando \marginsize
	%\usepackage[papersize={22cm,27cm},lmargin=2cm, rmargin=2cm,top=2cm,bottom=2cm]{geometry}
	\usepackage[papersize={22cm,27cm}]{geometry}
	\geometry{left=2cm, right=2cm, top=2cm, bottom=2cm}

	\newcommand{\bibliografia}{\addcontentsline{toc}{section}{Bibliografía}}
\fi







\newcommand{\signature}{\underline{\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,}}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Idioma español y carácteres utf8x
\usepackage[utf8]{inputenc}
\usepackage{url}

\usepackage{multicol}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tamaño de fuente
% \usepackage{anyfontsize}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figuras y tablas
\usepackage[pdftex]{graphicx} % PDFLaTeX
\usepackage{subfigure}
\usepackage{float}
\usepackage{tikz}
\usepackage{multirow}
% \DeclareGraphicsExtensions{.png,.pdf,.jpg}

\usepackage{comment}
\usepackage{color} % http://latexcolor.com/
\definecolor{gray}{rgb}{0.5, 0.5, 0.5}


\usepackage[spanish, es-lcroman, es-tabla]{babel}
\usepackage{enumerate}
\usepackage{appendix}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tamaño de la fuente
% \usepackage{anyfontsize}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fuente matemática
\usepackage{amssymb}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tablas
% \usepackage{tabulary}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estilo de página
\usepackage{fancyhdr}
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
% Bibliografía
\usepackage[round]{natbib}
%\usepackage[style=apa]{biblatex}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% indentacion
\usepackage{indentfirst}
\setlength{\parindent}{18pt}
\setlength{\parskip}{3pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% spacing
\usepackage{setspace}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% references
\usepackage{varioref}

%Definiciones
\newtheorem{defin}{Definición}
\newtheorem{teorema}{Teorema}
\newtheorem{obs}{Observación}
\setlength{\parindent}{2cm}                                             % Sangrado
\renewcommand{\baselinestretch}{1.5}                                    % Espaciado 1.5
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
	\listoftables
	\listoffigures
	\newpage
}



\if@dos
	\mode<presentation>{}
	\setbeamercovered{transparent}
	\useinnertheme{default}
	\usefonttheme{professionalfonts}
	\setbeamercolor{framesubtitle}{fg=azul}
	\setbeamercolor{frametitle}{fg=azul}
	\setbeamercolor{title}{fg=azul}


	\textwidth=0.75\paperwidth
	\setbeamertemplate{navigation bar}{}
\fi

\if@escrito
\makeatletter
\def \maketitle {%
	\newgeometry{left=4cm, right=2.5cm, top=4cm, bottom=2.5cm}
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
				%\includegraphics[scale=0.06]{img/usach.psd}
				\scalebox{0.3}{\inegrotrz}
				%\scalebox{0.7}{\inegrotrz}
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
        {\fontsize{10}{1} Memoria para obtener el título de Analista en Computación Científica}
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
%		{\fontsize{12}{1} {\bf \@tema}}\\
%		\vspace{1em}
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

%	\vfill
%	\begin{center}
%		{\fontsize{10}{1} Santiago, Chile\\ \the\year}
%	\end{center}
	\newpage
	\restoregeometry
}
\makeatother
\fi












\usepackage{lipsum}
\makeatletter
	\renewcommand\lips@dolipsum{%
		\ifnum\value{lips@count}<\lips@max\relax
			\addtocounter{lips@count}{1}%
			\csname lipsum@\romannumeral\c@lips@count\endcsname
			\lips@dolipsum
		\fi
	}
\makeatother


\newenvironment{descripcion}{
	\begingroup\color{gray}
} {
	\endgroup
}
\newenvironment{arreglar}{
	\begingroup\color{red}
} {
	\endgroup
}
\newenvironment{falta}{
	\begingroup\color{red}
	{\bf {\large Falta}}
} {
	\endgroup
}
