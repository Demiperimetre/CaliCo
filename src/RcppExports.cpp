// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// MetropolisHastingsCpp
List MetropolisHastingsCpp(int Ngibbs, int Nmh, arma::vec theta_init, arma::vec r, arma::mat SIGMA, arma::vec Yf, arma::vec binf, arma::vec bsup, Function LogTest, int stream);
RcppExport SEXP _CaliCo_MetropolisHastingsCpp(SEXP NgibbsSEXP, SEXP NmhSEXP, SEXP theta_initSEXP, SEXP rSEXP, SEXP SIGMASEXP, SEXP YfSEXP, SEXP binfSEXP, SEXP bsupSEXP, SEXP LogTestSEXP, SEXP streamSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type Ngibbs(NgibbsSEXP);
    Rcpp::traits::input_parameter< int >::type Nmh(NmhSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type r(rSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type SIGMA(SIGMASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Yf(YfSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type binf(binfSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type bsup(bsupSEXP);
    Rcpp::traits::input_parameter< Function >::type LogTest(LogTestSEXP);
    Rcpp::traits::input_parameter< int >::type stream(streamSEXP);
    rcpp_result_gen = Rcpp::wrap(MetropolisHastingsCpp(Ngibbs, Nmh, theta_init, r, SIGMA, Yf, binf, bsup, LogTest, stream));
    return rcpp_result_gen;
END_RCPP
}
// MetropolisHastingsCppD
List MetropolisHastingsCppD(int Ngibbs, int Nmh, arma::vec theta_init, arma::vec r, arma::mat SIGMA, arma::vec Yf, arma::vec binf, arma::vec bsup, Function LogTest, int stream);
RcppExport SEXP _CaliCo_MetropolisHastingsCppD(SEXP NgibbsSEXP, SEXP NmhSEXP, SEXP theta_initSEXP, SEXP rSEXP, SEXP SIGMASEXP, SEXP YfSEXP, SEXP binfSEXP, SEXP bsupSEXP, SEXP LogTestSEXP, SEXP streamSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type Ngibbs(NgibbsSEXP);
    Rcpp::traits::input_parameter< int >::type Nmh(NmhSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type r(rSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type SIGMA(SIGMASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Yf(YfSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type binf(binfSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type bsup(bsupSEXP);
    Rcpp::traits::input_parameter< Function >::type LogTest(LogTestSEXP);
    Rcpp::traits::input_parameter< int >::type stream(streamSEXP);
    rcpp_result_gen = Rcpp::wrap(MetropolisHastingsCppD(Ngibbs, Nmh, theta_init, r, SIGMA, Yf, binf, bsup, LogTest, stream));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_CaliCo_MetropolisHastingsCpp", (DL_FUNC) &_CaliCo_MetropolisHastingsCpp, 10},
    {"_CaliCo_MetropolisHastingsCppD", (DL_FUNC) &_CaliCo_MetropolisHastingsCppD, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_CaliCo(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
