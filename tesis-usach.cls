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

\newif\if@by
\DeclareOption{by}{\@bytrue}


\newif\ifpropuesta\propuestafalse
\DeclareOption{propuesta}{\propuestatrue}
\ProcessOptions\relax

\if@escrito
	\LoadClass[10pt, oneside]{book}
	\RequirePackage{titlesec}
	\titleformat{\chapter}[display]{\normalfont\bfseries}{\chaptertitlename\ \thechapter}{10pt}{}
	\titlespacing*{\chapter}{0pt}{0pt}{10pt}
	\titleformat{\section}[hang]{\bf}{{\bf \thesection\enspace}}{10pt}{}
	\titleformat{\subsection}[hang]{\bf}{{\bf \thesubsection\enspace}}{10pt}{}
	\titleformat{\subsubsection}[hang]{\bf}{{\bf \thesubsubsection\enspace}}{10pt}{}
	\setcounter{secnumdepth}{3}
	\setcounter{tocdepth}{3}

	\newcommand{\bibliografia}{\addcontentsline{toc}{chapter}{Bibliografía}}
\fi

\if@inf
	\LoadClass[12pt, oneside]{article}
	\RequirePackage{titlesec}
	\titleformat{\section}[hang]{\bf}{{\bf \thesection\enspace}}{14pt}{}
	\titleformat{\subsection}[hang]{\bf}{{\bf \thesubsection\enspace}}{14pt}{}
	\titleformat{\subsubsection}[hang]{\bf}{{\bf \thesubsubsection\enspace}}{14pt}{}
	\setcounter{secnumdepth}{3}
	\setcounter{tocdepth}{3}

	\newcommand{\bibliografia}{\addcontentsline{toc}{section}{Bibliografía}}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tamaño de la hoja
\usepackage{anysize} % Soporte para el comando \marginsize
\usepackage[papersize={21.6cm, 27.9cm}]{geometry}
\geometry{left=4cm, top=2.5cm, right=2.5cm, bottom=2.5cm}%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bibliografía y referencias
%\RequirePackage[natbibapa]{apacite}
\RequirePackage{hyperref}
\RequirePackage[apaciteclassic]{apacite}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Idioma español y carácteres utf8x
\usepackage[utf8]{inputenc}
\usepackage{url}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figuras y tablas
\usepackage[pdftex]{graphicx} % PDFLaTeX
\usepackage{float}
\usepackage{multicol}
\usepackage{multirow}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\usepackage{comment}
\usepackage{color} % http://latexcolor.com/
\definecolor{gray}{rgb}{0.5, 0.5, 0.5}


\usepackage[spanish, es-lcroman, es-tabla, es-noquoting]{babel}
\usepackage{enumerate}
\usepackage{appendix}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fuente matemática
\usepackage{amssymb}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identación y spacing
\usepackage{setspace}
\usepackage{indentfirst}
\setlength{\parindent}{18pt}
\setlength{\parskip}{3pt}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% references
\usepackage{varioref}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lorem ipsum
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Para crear argumentos para los comandos
\RequirePackage{xkeyval}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interlineado
\renewcommand{\baselinestretch}{1.5}







\newcommand{\signature}{\underline{\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,}}







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














\makeatletter
\def \aditionalpages {%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% COPYRIGHT
    %\setlength{\unitlength}{1cm} %Especificar unidad de trabajo
    %\thispagestyle{empty}
    %\vspace*{18cm}
    %\hspace*{-2.0cm}$\copyright$ \@author\\
    %Queda prohibida la reproducción parcial o total de esta obra en cualquier forma, medio o procedimiento sin permiso por escrito del autor.
    %\newpage




	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% PRIMERA HOJA
	\setcounter{page}{1}
	\setlength{\unitlength}{1cm} %Especificar unidad de trabajo
	\begin{center}
		{\fontsize{14}{1} {\bf \@title}}
	\end{center}
	\vfill
	\begin{center}
		{\fontsize{12}{1} Trabajo de Titulación presentado a la \facultad, en cumplimiento a los requisitos exigidos para optar al título de Analista Computacional Científico.}
	\end{center}
	\vfill
	\begin{center}
		{\fontsize{14}{1} \usach\\Santiago, Chile\\ \the\year}
	\end{center}
	\newpage

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% SEGUNDA HOJA
	\begin{figure}[h]
		\begin{minipage}{0.70\textwidth}
			\begin{flushleft}
				{\fontsize{12}{1} \usach}\\
				{\fontsize{12}{1} \facultad}\\
				{\fontsize{12}{1} \depto}
			\end{flushleft}
		\end{minipage}
		\begin{minipage}{0.15\textwidth}
			\begin{flushright}
				%\includegraphics[scale=0.05]{img/usach.png}
				\scalebox{0.17}{\input{img/logo}}
			\end{flushright}
		\end{minipage}
	\end{figure}
    \vspace{1em}%\vfill
	\begin{center}
		{\fontsize{14}{1} {\bf \@title}}
	\end{center}
	\vspace{1em}%\vfill
	\begin{center}
		\begin{tabular}{cc}
			\@author\\
		\end{tabular}
	\end{center}

    \vspace{1em}
    El presente trabajo se desarrolló en el Laboratorio de Computación del \depto\ (DMCC) de la \usach\ (USACH). Bajo la supervisión del profesor guía \@prof\ y ha sido aprobado por los miembros de la comisión calificadora \@revisorU\ y \@revisorD.

    \hspace*{-2.5cm}
    \begin{minipage}[b]{0.65\textwidth}
    \begin{table}[H]
	    \begin{tabular}{c}
		    \fontsize{12}{2} \signature\\
		    \\
		    \fontsize{12}{2} \@director\\
		    \fontsize{12}{2} Director
		    \end{tabular}
    \end{table}
    \end{minipage}
    \begin{minipage}[b]{0.5\textwidth}
	\vspace*{1cm}
    \begin{table}[H]
	    \begin{tabular}{c}
		    \fontsize{12}{2} \signature\\
		    \\
		    \fontsize{12}{2} \@prof\\
		    \fontsize{12}{2} Profesor Guía\\
		    \\
		    \end{tabular}
    \end{table}\bigskip
    \begin{table}[H]
	    \begin{tabular}{c}
		    \fontsize{12}{2} \signature\\
		    \\
		    \fontsize{12}{2} \@revisorU\\
		    \fontsize{12}{2} Profesor Informante\\
		    \\
		    \end{tabular}
    \end{table}\bigskip
    \begin{table}[H]
	    \begin{tabular}{c}
		    \fontsize{12}{2} \signature\\
		    \\
		    \fontsize{12}{2} \@revisorD\\
		    \fontsize{12}{2} Profesor Informante
		    \end{tabular}
    \end{table}
    \end{minipage}
	\newpage
}
\makeatother

\newcommand{\makecopyright}[1]{
	\null
	\vfill
    \ifpropuesta
        % no va copyright
    \else
        \begin{flushleft}
		\copyright\;
			\@author - \the\year\\
			\if@by
			\scalebox{0.3}{\input{img/by}}
			~$\bullet$~~ Algunos derechos reservados. Esta obra está bajo una Licencia Creative Commons Atribución-Chile 3.0. Sus condiciones de uso pueden ser revisadas en: \url{http://creativecommons.org/licenses/by/3.0/cl/}.
			\fi
        \end{flushleft}
    \fi
}

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


%\begin{figure}[H]
%	\centering
%	\scalebox{0.1}{\input{img/neurona}}
%	\caption{Neurona}
%	\label{fig:neurona}
%\end{figure}

\RequirePackage{xkeyval}
\makeatletter
\define@key{tikzimg}{scale}{\def\escala{#1}}
\makeatother
%\define@key{imagen}{b}{\renewcommand*\verbsetb{#1}}

\newenvironment{imagen}{
\setkeys{tikzimg}{scale=1.0}
	\begin{figure}[H]
	\centering
} {
	\end{figure}
}
