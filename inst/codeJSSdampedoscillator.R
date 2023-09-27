## code for exemple in JSS file
n <- 50
t <- seq(0,2,length.out=n)
code <- function(t,theta)
{
  w0 <- sqrt(theta[3]/theta[4])
  return(theta[1]*exp(-theta[2]*w0*t)*sin(sqrt(1-theta[2]^2)*w0*t+theta[5]))
}


