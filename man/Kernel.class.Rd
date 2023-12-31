% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Kernel.R
\name{Kernel.class}
\alias{Kernel.class}
\title{A Reference Class to generates differents model objects}
\description{
See the function \code{\link{model}} which produces an instance of this class
}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{code}}{a function which takes in entry X and theta}

\item{\code{X}}{the matrix of the forced variables}

\item{\code{var}}{the variance for the covariance function}

\item{\code{psi}}{the scale vector of correlation length}

\item{\code{n}}{number of experiments}

\item{\code{Kernel.type}}{the chosen form of covariance}

\item{\code{Cov}}{the covariance}
}
\if{html}{\out{</div>}}
}
\section{Active bindings}{
\if{html}{\out{<div class="r6-active-bindings">}}
\describe{
\item{\code{code}}{a function which takes in entry X and theta}

\item{\code{n}}{number of experiments}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-Kernel.class-new}{\code{Kernel.class$new()}}
\item \href{#method-Kernel.class-clone}{\code{Kernel.class$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Kernel.class-new"></a>}}
\if{latex}{\out{\hypertarget{method-Kernel.class-new}{}}}
\subsection{Method \code{new()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Kernel.class$new(X = NA, var = NA, psi = NA, Kernel.type = NA)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-Kernel.class-clone"></a>}}
\if{latex}{\out{\hypertarget{method-Kernel.class-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Kernel.class$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
