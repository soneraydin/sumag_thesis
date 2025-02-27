# setwd("/home/soneraydin/Masaüstü/TR_New/author_clustering")
setwd("/home/soneraydin/Masaüstü/MAG_TR_New_Data")

# Function to initialize W and H matrices randomly
rand_init <- function(filename, p) {
  d <- read.table(filename, header=F)
  nrows <- d[1,1]; ncols <- d[1,2]; nratings <- d[1,3]
  w <- matrix(runif(nrows * p) * 2/p , nrows, p); h <- matrix(runif(ncols * p) * 2/p, p, ncols)
  initial_factors <- list(W=w, H=h)
  return(initial_factors)
}

# SGD for coupled matrix factorization
sgd_cmf <- function(filename1, filename2, p, samplesize, alpha, maxiter){
  X <- read.table(filename1, header=F, skip=1)
  Y <- read.table(filename2, header=F, skip=1)
  test_sample_x <- sample(1:nrow(X), 100000, replace=F); X <- X[X[,3]!=0,]
  x_test <- X[test_sample_x,]; X <- X[-test_sample_x,]
  factors_x <- rand_init(filename1, p); W <- factors_x$W; H <- factors_x$H
  factors_y <- rand_init(filename2, p); M <- factors_y$H
  for(t in 1:maxiter){
    sample_indices_x <- sample(1:nrow(X), samplesize, replace=F)
    sample_indices_y <- sample(1:nrow(Y), samplesize, replace=F)
    y2 <- as.numeric(X[sample_indices_x,1] %in% Y[,1])
    cat("Iteration-", t, "\n")
    for(n in 1:samplesize){
      nx <- sample_indices_x[n]
      ny <- sample_indices_y[n]
      ix <- X[nx,1]; jx <- X[nx,2]; xij <- X[nx,3]
      iy <- Y[ny,1]; jy <- Y[ny,2]; yij <- Y[ny,3]
      for(k in 1:p){
        W[ix,k] <- W[ix,k] + 2*alpha*(xij - W[ix,]%*%H[,jx])*H[k,jx] + 
                             2*alpha*(y2[n] - W[ix,]%*%M[,jx])*M[k,jx]
        H[k,jx] <- H[k,jx] + 2*alpha*(xij - W[ix,]%*%H[,jx])*W[ix,k]
        M[k,jy] <- M[k,jy] + 2*alpha*(yij - W[iy,]%*%M[,jy])*W[iy,k]
      }
    }
  }
  # Mean absolute percentage error calculation for the overall result
  # MAPE for test sets
  mape <- test_mape(x_test, W, H)
  cat("MAPE for testing set:", mape, "\n")
  factors <- list()
  factors$W <- W; factors$H <- H; factors$M <- M
  return(factors)
}

# Regularized version of the coupled matrix factorization
regularized_sgd_cmf <- function(filename1, filename2, p, samplesize, 
                                    alpha, lambda1, lambda2, lambda3, maxiter){
  X <- read.table(filename1, header=F, skip=1)
  Y <- read.table(filename2, header=F, skip=1)
  test_sample_x <- sample(1:nrow(X), 100000, replace=F); X <- X[X[,3]!=0,]
  x_test <- X[test_sample_x,]; X <- X[-test_sample_x,]
  factors_x <- rand_init(filename1, p); W <- factors_x$W; H <- factors_x$H
  factors_y <- rand_init(filename2, p); M <- factors_y$H
  for(t in 1:maxiter){
    sample_indices_x <- sample(1:nrow(X), samplesize, replace=F)
    sample_indices_y <- sample(1:nrow(Y), samplesize, replace=F)
    y2 <- as.numeric(X[sample_indices_x,1] %in% Y[,1])
    cat("Iteration-", t, "\n")
    for(n in 1:samplesize){
      nx <- sample_indices_x[n]
      ny <- sample_indices_y[n]
      ix <- X[nx,1]; jx <- X[nx,2]; xij <- X[nx,3]
      iy <- Y[ny,1]; jy <- Y[ny,2]; yij <- Y[ny,3]
      for(k in 1:p){
        W[ix,k] <- W[ix,k] + 2*alpha*(xij - W[ix,]%*%H[,jx])*H[k,jx] + 
                             2*alpha*(y2[n] - W[ix,]%*%M[,jx])*M[k,jx] - 2*alpha*lambda1*W[ix,k]
        H[k,jx] <- H[k,jx] + 2*alpha*(xij - W[ix,]%*%H[,jx])*W[ix,k] - 2*alpha*lambda2*H[k,jx]
        M[k,jy] <- M[k,jy] + 2*alpha*(yij - W[iy,]%*%M[,jy])*W[iy,k] - 2*alpha*lambda3*M[k,jy]
      }
    }
  }
  # Mean absolute percentage error calculation for the overall result
  error <- 0
  for(n in 1:nrow(X)){
    i <- X[n,1]; j <- X[n,2]; xij <- X[n,3]
    error <- error + abs(xij - W[i,]%*%H[,j])/xij
  }
  mape <- 100*error/nrow(X); cat("MAPE for the training set: ",mape,"\n")
  # MAPE for test sets
  mape <- test_mape(x_test, W, H)
  cat("MAPE for testing set:", mape, "\n")
  factors <- list()
  factors$W <- W; factors$H <- H; factors$M <- M
  return(factors)
}

# Mean absolute percentage error calculation for testing set
test_mape <- function(X, W, H) {
  error <- 0
  for(n in 1:nrow(X)){
    i <- X[n,1]; j <- X[n,2]; xij <- X[n,3]
    error <- error + abs(xij - W[i,]%*%H[,j])/xij
  }
  mape <- 100*error/nrow(X)
  return(mape)
}

# An example to demonstrate the regularized CMF
# Using author-paper and author-reference matrices as input
rfmat <- regularized_sgd_cmf("author_paper_matrix.txt", "author_reference_matrix.txt",
                                 19, 10000, 0.05, 0.0025, 0.0025, 0.0025, 2000)

