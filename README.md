# PCORcpp

This package provides a `C++` implementation of Projection Correlation (PCOR) computations, leveraging `Rcpp`, `Armadillo` and `OPENMP` for performance optimization An `R` implementation is available at [PCOR](https://github.com/Yilin-Zhang10/PCOR).

## Installation

```r
remotes::install_github("chia202/PCORcpp")
```

## Features

- `pcov()`: Compute projection covariance between two random vectors
- `pcor()`: Compute projection correlation coefficient between two random vectors  
- `pcor.test()`: Perform projection correlation independence test

## Usage Example

```r
library(PCORcpp)

set.seed(123)
X <- matrix(rnorm(100 * 5), 100, 5)
Y <- matrix(rnorm(100 * 3), 100, 3)

# Compute projection covariance
result <- pcov(X, Y, n.threads = 4)
print(result)

# Compute projection correlation coefficient
result <- pcor(X, Y, estimation.method = "u", n.threads = 4)
print(result)

# Perform independence test
test_result <- pcor.test(X, Y, times = 199, n.threads = 4)
print(test_result)
```

## Parameters

- `estimation.method`: Estimation method, "u" for U-statistic, "v" for V-statistic
- `times`: Number of permutations for permutation test
- `n.threads`: Number of threads for parallel computation

## Reference

1. L. Zhu, K. Xu, R. Li, W. Zhong (2017). Projection correlation between two random vectors. Biometrika, 104, 829-843.
