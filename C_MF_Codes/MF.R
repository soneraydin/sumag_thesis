setwd("/home/soneraydin/Masaüstü/MAG_TR_New_Data")

library(Matrix)

# Function to initialize W and H matrices randomly
rand_init <- function(filename, p) {
  d <- read.table(filename, header=F)
  nrows <- d[1,1]; ncols <- d[1,2]; nratings <- d[1,3]
  w <- matrix(runif(nrows * p) * 2/p , nrows, p); h <- matrix(runif(ncols * p) * 2/p, p, ncols)
  initial_factors <- list(W=w, H=h)
  return(initial_factors)
}

# Stochastic gradient descent implementation for the matrix factorization
# When 'samplesize' argument of the function is greater than 1, it works as mini-batch SGD
sgd_mf <- function(filename, p, samplesize, alpha0, maxiter){
  alpha <- alpha0
  X <- read.table(filename, header=F, skip=1); X <- X[X[,3]!=0,]
  test_sample <- sample(1:nrow(X), 100000, replace=F)
  x_test <- X[test_sample,]; X <- X[-test_sample,]
  factors <- wh_init(filename, p); W <- factors$W; H <- factors$H
  for(t in 1:maxiter){
    sample_indices <- sample(1:nrow(X), samplesize, replace=F)
    cat("Iteration-", t, "\n")
    for(n in sample_indices){
      i <- X[n,1]; j <- X[n,2]; xij <- X[n,3]
      for(k in 1:p){
        W[i,k] <- W[i,k] + 2*alpha*(xij - W[i,]%*%H[,j])*H[k,j]
        H[k,j] <- H[k,j] + 2*alpha*(xij - W[i,]%*%H[,j])*W[i,k]
      }
    }
  }
  # Mean absolute percentage error calculation for the overall result
  error <- 0
  for(n in 1:nrow(X)){
    i <- X[n,1]; j <- X[n,2]; xij <- X[n,3]
    error <- error + abs(xij - W[i,]%*%H[,j])/xij
  }
  mape <- 100*error/nrow(X); cat("Mean absolute percentage error for training set: ",mape,"\n")
  test_mape(x_test, W, H)
  factors$W <- W; factors$H <- H
  return(factors)
}

# Regularized version of the matrix factorization function
regularized_sgd_mf <- function(filename, p, samplesize, alpha0, beta, maxiter){
  alpha <- alpha0
  X <- read.table(filename, header=F, skip = 1); X <- X[X[,3]!=0,]
  test_sample <- sample(1:nrow(X), 100000, replace=F)
  x_test <- X[test_sample,]; X <- X[-test_sample,]
  factors <- rand_init(filename, p); W <- factors$W; H <- factors$H
  for(t in 1:maxiter){
    sample_indices <- sample(1:nrow(X), samplesize, replace=F)
    cat("Iteration-", t, "\n")
    for(n in sample_indices){
      i <- X[n,1]; j <- X[n,2]; xij <- X[n,3]
      for(k in 1:p){
        W[i,k] <- W[i,k] + 2*alpha*(xij - W[i,]%*%H[,j])*H[k,j] - 2*alpha*beta*W[i,k]
        H[k,j] <- H[k,j] + 2*alpha*(xij - W[i,]%*%H[,j])*W[i,k] - 2*alpha*beta*H[k,j]
      }
    }
  }
  # Mean absolute percentage error calculation for the overall result
  error <- 0
  for(n in 1:nrow(X)){
    i <- X[n,1]; j <- X[n,2]; xij <- X[n,3]
    error <- error + abs(xij - W[i,]%*%H[,j])/xij
  }
  mape <- 100*error/nrow(X); cat("Mean absolute percentage error for training set: ",mape,"\n")
  test_mape(x_test, W, H)
  factors$W <- W; factors$H <- H
  return(factors)
}

# Mean absolute percentage error calculation for testing set
test_mape <- function(X, W, H) {
  error <- 0
  for(n in 1:nrow(X)){
    i <- X[n,1]; j <- X[n,2]; xij <- X[n,3]
    error <- error + abs(xij - W[i,]%*%H[,j])/xij
  }
  mape <- 100*error/nrow(X); cat("Mean absolute percentage error for testing set: ",mape,"\n")
}

# Example with author-paper matrix 
# apf <- sgd_mf("author_paper_matrix.txt", 19, 50000, 0.3, 10)

rapf <- regularized_sgd_mf("author_paper_matrix.txt", 19, 10000, 0.05, 0.0025, 2000)

