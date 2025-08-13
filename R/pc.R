#' @name pcov
#' @title Calculate the projection covariance of two random vectors
#'
#' @description This function computes the projection covariance between two random vectors.
#'
#' @param X A numeric matrix of dimension n x p, where each row is an i.i.d. observation of the first vector.
#' @param Y A numeric matrix of dimension n x q, where each row is an i.i.d. observation of the second vector.
#' @param estimation.method A character string, either "u" or "v". If "u", U-statistics are used, if "v", V-statistics are used.
#' @param n.threads An integer specifying the number of threads to use for parallel computation. Default is 4.
#'
#' @return A numeric scalar representing the projection covariance between X and Y.
#'
#' @examples
#' X <- matrix(rnorm(100 * 34, 1), 100, 34)
#' Y <- matrix(rnorm(100 * 62, 1), 100, 62)
#' pcov(X, Y, estimation.method = "u", n.threads = 4)
#'
#' @export
pcov <- function(X, Y, estimation.method = "u", n.threads = 4) {
    pcov_cpp(X, Y, estimation.method, n.threads)
}

#' @name pcor
#' @title Calculate the projection correlation of two random vectors
#'
#' @description This function computes the projection correlation between two random vectors.
#'
#' @param X A numeric matrix of dimension n x p, where each row is an i.i.d. observation of the first vector.
#' @param Y A numeric matrix of dimension n x q, where each row is an i.i.d. observation of the second vector.
#' @param estimation.method A character string, either "u" or "v". If "u", U-statistics are used, if "v", V-statistics are used.
#' @param n.threads An integer specifying the number of threads to use for parallel computation. Default is 4.
#'
#' @return A numeric scalar representing the projection correlation between X and Y.
#'
#' @examples
#' X <- matrix(rnorm(100 * 34, 1), 100, 34)
#' Y <- matrix(rnorm(100 * 62, 1), 100, 62)
#' pcor(X, Y, estimation.method = "u", n.threads = 4)
#'
#' @export
pcor <- function(X, Y, estimation.method = "u", n.threads = 4) {
    pcor_cpp(X, Y, estimation.method, n.threads)
}

#' @name pcor.test
#' @title Perform projection correlation independence test
#'
#' @description This function performs a projection correlation independence test between two random vectors.
#'
#' @param X A numeric matrix of dimension n x p, where each row is an i.i.d. observation of the first vector.
#' @param Y A numeric matrix of dimension n x q, where each row is an i.i.d. observation of the second vector.
#' @param estimation.method A character string, either "u" or "v". If "u", U-statistics are used, if "v", V-statistics are used.
#' @param times An integer specifying the number of permutations to perform. Default is 199.
#' @param n.threads An integer specifying the number of threads to use for parallel computation. Default is 4.
#'
#' @return A list with components:
#' \item{method}{The method name: \code{"Projection Correlation Permutation Test of Independence"}}
#' \item{stat.value}{The test statistic value (scaled by sample size).}
#' \item{p.value}{The p-value computed from the permutation distribution under the null hypothesis of independence.}
#'
#' @examples
#' X <- matrix(rnorm(10 * 7, 1), 10, 7)
#' Y <- matrix(rnorm(10 * 6, 1), 10, 6)
#' pcor.test(X, Y, estimation.method = "v", times = 199, n.threads = 4)
#'
#' @export
pcor.test <- function(X, Y, estimation.method = "u", times = 199, n.threads = 4) {
    pcor_test_cpp(X, Y, estimation.method, times, n.threads)
}
