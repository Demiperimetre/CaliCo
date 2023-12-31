% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{MetropolisHastingsCppD}
\alias{MetropolisHastingsCppD}
\title{C++ implementation of the algorithm for parameter calibration (with discrepancy)}
\usage{
MetropolisHastingsCppD(
  Ngibbs,
  Nmh,
  theta_init,
  r,
  SIGMA,
  Yf,
  binf,
  bsup,
  LogTest,
  stream
)
}
\arguments{
\item{Ngibbs}{the number of iteration in the Metropolis within Gibbs}

\item{Nmh}{the number of iteration in the Metropolis Hastings}

\item{theta_init}{the starting point}

\item{r}{regulation percentage in the modification of the k}

\item{SIGMA}{the covariance of the proposition distribution}

\item{Yf}{the vector of recorded data}

\item{binf}{the lower bound of the parameters to calibrate}

\item{bsup}{the upper bound of the parameters to calibrate}

\item{LogTest}{the log posterior density distribution}

\item{stream}{(default=1) if stream=0 the progress bar is disabled}
}
\value{
list of outputs: \itemize{
\item PHIwg the points of the Metropolis within Gibbs algorithm in the transformed space
\item PHI the points of the Metropolis Hastings algorithm in the transformed space
\item THETAwg the points of the Metropolis within Gibbs algorithm in the real space
\item THETA the points of the Metropolis Hastings algorithm in the real space
\item AcceptationRatioWg the vector of the acceptance ratio for each parameter in the Metropolis within Gibbs
\item AcceptationRatio the acceptance ratio in the Metropolis Hastings
\item S the covariance computed after the Metropolis within Gibbs
\item LikeliWG the likelihood computed at each iteration of the Metropolis within Gibbs algorithm
\item Likeli the likelihood computed at each iteration of the Metropolis Hastings algorithm
 }
}
\description{
Run a Metropolis Hastings within Gibbs algorithm and a Metropolis Hastings algorithm with the covariance matrix estimated on the
the sample set generated in the Metropolis within Gibbs. This algorithm is suitable only for models with discrepancy.
}
