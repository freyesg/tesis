\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ppt}[2015/12/21 Presentacion USACH]

\newif\if@uno
\DeclareOption{ppt1}{\@unotrue}

\newif\if@dos
\DeclareOption{ppt2}{\@dostrue}

\newif\if@tres
\DeclareOption{ppt3}{\@trestrue}

\ProcessOptions\relax
\LoadClass[8pt]{beamer}


%\usepackage[scaled]{helvet}


\RequirePackage{tikz}
\RequirePackage[spanish]{babel}
\RequirePackage[utf8]{inputenc}
\RequirePackage{amssymb}
\RequirePackage{natbib}

\definecolor{naranjo}{cmyk}{0.0, 0.55, 0.90, 0.0}
\definecolor{gris}{cmyk}{0, 0, 0, 0.4}
\definecolor{azul}{cmyk}{1, 0.6, 0.0, 0.2}

\setbeamercolor{block title}{use=structure,fg=white,bg=azul}
\setbeamercolor{block body}{use=structure,fg=black,bg=gris!40}
\setbeamertemplate{blocks}[rounded]%[shadow=true]

%\useoutertheme{infolines}
\setbeamercovered{transparent}
\useinnertheme{default}
\usefonttheme{professionalfonts}
\mode<presentation>{}

\setbeamertemplate{footline}[frame number]

\definecolor{c231f20}{RGB}{35,31,32}
\definecolor{ced1c24}{RGB}{237,28,36}
\definecolor{cc29438}{RGB}{194,148,56}
\definecolor{c263d96}{RGB}{38,61,150}
\definecolor{c165096}{RGB}{22,80,150}
\definecolor{cbdbcbc}{RGB}{189,188,188}
\definecolor{cffffff}{RGB}{255,255,255}
\definecolor{cfff200}{RGB}{255,242,0}
\definecolor{cf57b25}{RGB}{245,123,37}
\definecolor{c918f90}{RGB}{145,143,144}
\definecolor{ca7a5a6}{RGB}{167,165,166}


\newcommand{\iazltrz}{\input{img/iazltrz}}
\newcommand{\ibcotrz}{\input{img/ibcotrz}}
\newcommand{\icolorazl}{\input{img/icolorazl}}
\newcommand{\icolorbco}{\input{img/icolorbco}}
\newcommand{\inegrotrz}{\input{img/inegrotrz}}
\newcommand{\icolornrj}{\input{img/icolornrj}}
\newcommand{\usachnrj}{\input{img/usach}}

\if@uno
	\setbeamercolor{framesubtitle}{fg=azul}
	\setbeamercolor{frametitle}{fg=azul}
	\setbeamercolor{title}{fg=azul}

	\newcommand{\Ag}{(\paperwidth*0.831, \paperheight*0.5)}
	\newcommand{\Cg}{(\paperwidth*0.783,	-\paperheight*0.499)}

	\newcommand{\An}{(\paperwidth*0.854, \paperheight*0.5)}
	\newcommand{\Cn}{(\paperwidth*0.7575,	-\paperheight*0.499)}

	\newcommand{\controlcinco}{(\paperwidth*0.824, -\paperheight*0.2)}
	\newcommand{\controlseis}{(\paperwidth*0.883, \paperheight*0.23)}

	\newcommand{\controltres}{(\paperwidth*0.871, \paperheight*0.28)}
	\newcommand{\controlcuatro}{(\paperwidth*0.847, -\paperheight*0.12)}


	\usebackgroundtemplate
	{
		\makebox[\paperwidth][c]{
			\begin{tikzpicture}
				\draw[black!0] (0, 0) circle (1pt); % Punto (0, 0) de referencia
				\draw[gris!80] \Ag .. controls \controltres and \controlcuatro .. \Cg;
				\draw[naranjo] \Cn .. controls \controlcinco and \controlseis .. \An;
				\node[inner sep=0pt, below](russell) at (\paperwidth*0.922, \paperheight*0.476) {\scalebox{0.153}{\icolornrj}};
			\end{tikzpicture}
		}
	}
	\textwidth=0.75\paperwidth
	%\setbeamertemplate{navigation bar}{}
\fi

\if@dos
	\setbeamercolor{framesubtitle}{fg=azul}
	\setbeamercolor{frametitle}{fg=azul}
	\setbeamercolor{title}{fg=azul}

	\newcommand{\uno}{(-\paperwidth*0.5, \paperheight*0.5)}
	\newcommand{\dos}{(\paperwidth*0.5, \paperheight*0.5)}
	\newcommand{\tres}{(\paperwidth*0.5, -\paperheight*0.5)}
	\newcommand{\cuatro}{(-\paperwidth*0.5, -\paperheight*0.5)}

	\newcommand{\A}{(\paperwidth*0.845, \paperheight*0.5)}
	\newcommand{\B}{(\paperwidth*0.996, \paperheight*0.5)}
	\newcommand{\C}{(\paperwidth*0.996,	-\paperheight*0.499)}
	\newcommand{\D}{(\paperwidth*0.899, -\paperheight*0.499)}

	\newcommand{\An}{(\paperwidth*0.84, \paperheight*0.5)}
	\newcommand{\Bn}{(\paperwidth*0.89,	-\paperheight*0.499)}
	\newcommand{\Cn}{(\paperwidth*0.884,	-\paperheight*0.499)}

	\newcommand{\controluno}{(\paperwidth*0.899+3, -\paperheight*0.16)}
	\newcommand{\controldos}{(\paperwidth*0.872+7, \paperheight*0.21)}
	\newcommand{\controltres}{(\paperwidth*0.888, \paperheight*0.2)}
	\newcommand{\controlcuatro}{(\paperwidth*0.8999, -\paperheight*0.16)}

	\newcommand{\controlcinco}{(\paperwidth*0.8999, -\paperheight*0.16)}
	\newcommand{\controlseis}{(\paperwidth*0.88, \paperheight*0.19)}


	%\setbeamertemplate{background}
	\usebackgroundtemplate
	{
		\makebox[\paperwidth][c]{
			\begin{tikzpicture}
				\filldraw[red!0] (0, 0) circle (1pt); % Punto (0, 0) de referencia
				\filldraw[naranjo] \An .. controls \controltres and \controlcuatro .. \Bn -- \Cn .. controls \controlcinco and \controlseis .. \An -- cycle;
				\filldraw[azul] \A -- \B -- \C -- \D .. controls \controluno and \controldos .. \A -- cycle;
				%\node[inner sep=0pt, below](russell) at (\paperwidth*0.9323, \paperheight*0.4809) {\includegraphics[scale=0.155]{img/icolor_nrj.png}};
				\node[inner sep=0pt, below](russell) at (\paperwidth*0.9323, \paperheight*0.4809) {\scalebox{0.153}{\icolornrj}};
			\end{tikzpicture}
		}
	}
\fi

\if@tres
	\addtobeamertemplate{block begin}{\setlength{\textwidth}{0.9\textwidth}}{}
	\addtobeamertemplate{block begin}{\setlength{\textheight}{0.8\textheight}}{}

	\setbeamercolor{framesubtitle}{fg=white}
	\setbeamercolor{frametitle}{fg=white}
	\setbeamercolor{title}{fg=azul}

	\newcommand{\pa}{(-\paperwidth*0.5, \paperheight*0.5)} % superior izquierdo
	\newcommand{\pb}{(-\paperwidth*0.5, \paperheight*0.366)} % inferior izquierdo
	\newcommand{\pc}{(\paperwidth*0.5,	\paperheight*0.291)} % inferior derecho
	\newcommand{\pd}{(\paperwidth*0.5,	\paperheight*0.5)} % superior derecho
	\newcommand{\pci}{(-\paperwidth*0.2, \paperheight*0.38)}
	\newcommand{\pcd}{(\paperwidth*0.25, \paperheight*0.35)}


	\newcommand{\an}{(-\paperwidth*0.5, \paperheight*0.353)} % superior izquierdo
	\newcommand{\bn}{(-\paperwidth*0.5, \paperheight*0.343)} % inferior izquierdo
	\newcommand{\cn}{(\paperwidth*0.5,	\paperheight*0.285)} % derecho
	\newcommand{\pcnupi}{(\paperwidth*0.19, \paperheight*0.35)} % punto de control superior derecho
	\newcommand{\pcnupii }{(-\paperwidth*0.22, \paperheight*0.37)} % punto de control superior izquierdo
	\newcommand{\pcndownd}{(\paperwidth*0.19, \paperheight*0.34)} % punto de control inferior izquierdo
	\newcommand{\pcndowndd }{(-\paperwidth*0.22, \paperheight*0.37)} % punto de control inferior derecho

	\setbeamertemplate{background canvas} {
		\makebox[\paperwidth][c]{
			\begin{tikzpicture}
				\filldraw[azul] \pa -- \pb .. controls \pci and \pcd .. \pc -- \pd -- \pa;
				\filldraw[naranjo] \an -- \bn .. controls \pcndowndd and \pcndownd .. \cn .. controls \pcnupi and \pcnupii .. \an -- cycle;
				\node[inner sep=0pt, below](russell) at (\paperwidth*0.4355, \paperheight*0.485) {\scalebox{0.153}{\icolornrj}};
				%\node[inner sep=0pt, below](russell) at (\paperwidth*0.4300, \paperheight*0.485) {\scalebox{0.153}{\usachnrj}};
				%\node[below of=russell] {\textup{UdeSantiago}};
			\end{tikzpicture}
		}
	}

\fi

%\setbeamertemplate{footline}
%{
%	\leavevmode%
%	\hbox{%
%	\begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{author in head/foot}%
		% \usebeamerfont{author in head/foot}\insertsection
%	\end{beamercolorbox}%
%	\begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
%		%\usebeamerfont{title in head/foot}\insertsubsection
%	\end{beamercolorbox}%
%	\begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
%		%\usebeamerfont{date in head/foot}
%		\insertframenumber{} / \inserttotalframenumber\hspace*{2ex}
%	\end{beamercolorbox}}%
%	\vskip0pt%
%}

\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{sections/subsections in toc}[square]
\setbeamertemplate{blocks}[rounded][shadow=true]
