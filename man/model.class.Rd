% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/models.R
\name{model.class}
\alias{model.class}
\title{A Reference Class to generates differents model objects}
\description{
See the function \code{\link{model}} which produces an instance of this class
This class comes with a set of methods, some of them being useful for the user:
See the documentation for \code{\link{model}}... Other methods
should not be called as they are designed to be used during the calibration process.


Fields should not be changed or manipulated by the user as they are updated internally
during the estimation process.
}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{code}}{a function which takes in entry X and theta}

\item{\code{X}}{the matrix of the forced variables}

\item{\code{Yexp}}{the experimental output}

\item{\code{n}}{the number of experiments}

\item{\code{d}}{the number of forced variables}

\item{\code{binf}}{the lower bound of the parameters for the DOE}

\item{\code{bsup}}{the upper bound of the parameters for the DOE}

\item{\code{opt.gp}}{a list of parameter for the surrogate (default NULL) \itemize{
\item{\strong{type}}{ type of the chosen kernel (value by default "matern5_2") from \code{\link{km}} function}
\item{\strong{DOE}}{ design of experiments for the surrogate (default value NULL)}}}

\item{\code{opt.emul}}{a list of parameter to establish the DOE (default NULL) \itemize{
\item{\strong{p}}{ the number of parameter in the model (default value 1)}
\item{\strong{n.emul}}{ the number of points for constituting the Design Of Experiments (DOE) (default value 100)}
\item{\strong{binf}}{ the lower bound of the parameter vector (default value 0)}
\item{\strong{bsup}}{ the upper bound of the parameter vector (default value 1)}}}

\item{\code{opt.sim}}{a list of parameter containing output of the code and corresponding DOE \itemize{
\item{\strong{Ysim}}{ Output of the code}
\item{\strong{DOEsim}}{ DOE corresponding to the output of the code}}}

\item{\code{model}}{the model choice (see \code{\link{model}} for more specification).}

\item{\code{opt.disc}}{a list of parameter for the discrepancy \itemize{
\item{\strong{kernel.type}}{ the kernel chosen for the Gaussian process}}}
}
\if{html}{\out{</div>}}
}
\section{Active bindings}{
\if{html}{\out{<div class="r6-active-bindings">}}
\describe{
\item{\code{binf}}{the lower bound of the parameters for the DOE}

\item{\code{bsup}}{the upper bound of the parameters for the DOE}

\item{\code{opt.gp}}{a list of parameter for the surrogate (default NULL) \itemize{
\item{\strong{type}}{ type of the chosen kernel (value by default "matern5_2") from \code{\link{km}} function}
\item{\strong{DOE}}{ design of experiments for the surrogate (default value NULL)}}}

\item{\code{opt.emul}}{a list of parameter to establish the DOE (default NULL) \itemize{
\item{\strong{p}}{ the number of parameter in the model (default value 1)}
\item{\strong{n.emul}}{ the number of points for constituting the Design Of Experiments (DOE) (default value 100)}
\item{\strong{binf}}{ the lower bound of the parameter vector (default value 0)}
\item{\strong{bsup}}{ the upper bound of the parameter vector (default value 1)}}}

\item{\code{opt.sim}}{a list of parameter containing output of the code and corresponding DOE \itemize{
\item{\strong{Ysim}}{ Output of the code}
\item{\strong{DOEsim}}{ DOE corresponding to the output of the code}}}

\item{\code{opt.disc}}{a list of parameter for the discrepancy \itemize{
\item{\strong{kernel.type}}{ the kernel chosen for the Gaussian process}}}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-model.class-new}{\code{model.class$new()}}
\item \href{#method-model.class-gglegend}{\code{model.class$gglegend()}}
\item \href{#method-model.class-plot}{\code{model.class$plot()}}
\item \href{#method-model.class-print}{\code{model.class$print()}}
\item \href{#method-model.class-discrepancy}{\code{model.class$discrepancy()}}
\item \href{#method-model.class-clone}{\code{model.class$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-model.class-new"></a>}}
\if{latex}{\out{\hypertarget{method-model.class-new}{}}}
\subsection{Method \code{new()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{model.class$new(code = NA, X = NA, Yexp = NA, model = NA)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-model.class-gglegend"></a>}}
\if{latex}{\out{\hypertarget{method-model.class-gglegend}{}}}
\subsection{Method \code{gglegend()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{model.class$gglegend()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-model.class-plot"></a>}}
\if{latex}{\out{\hypertarget{method-model.class-plot}{}}}
\subsection{Method \code{plot()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{model.class$plot(x, CI = "all", ...)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-model.class-print"></a>}}
\if{latex}{\out{\hypertarget{method-model.class-print}{}}}
\subsection{Method \code{print()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{model.class$print(...)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-model.class-discrepancy"></a>}}
\if{latex}{\out{\hypertarget{method-model.class-discrepancy}{}}}
\subsection{Method \code{discrepancy()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{model.class$discrepancy(theta, thetaD, var, X = self$X)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-model.class-clone"></a>}}
\if{latex}{\out{\hypertarget{method-model.class-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{model.class$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
