#include <RcppArmadillo.h>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

arma::mat sub_mat_cpp(const arma::mat &mat, int r, int n, int p)
{
	arma::rowvec ref = mat.row(r);
	arma::mat mat_ctr = mat.each_row() - ref;
	mat_ctr.shed_row(r); // remove the r-th row

	arma::mat mat_std;

	if (p > 1)
	{
		// normalize each row by its L2 norm
		arma::vec norms = sqrt(sum(square(mat_ctr), 1));
		mat_std = mat_ctr.each_col() / norms;
	}
	else if (p == 1)
	{
		arma::vec norms = abs(mat_ctr.col(0));
		mat_std = mat_ctr.each_col() / norms;
	}
	else
	{
		stop("p must be >= 1");
	}

	arma::mat A = arma::acos(mat_std * mat_std.t());

	// Replace NaN (result of acos out of domain) with 0
	A.elem(find_nonfinite(A)).zeros();

	return A;
}

arma::vec one_pcov_cpp(int r, const arma::mat &X, const arma::mat &Y, int n,
					   int p, int q, const std::string &estimation_method)
{
	arma::vec out(2);

	arma::mat mx = sub_mat_cpp(X, r, n, p);
	arma::mat my = sub_mat_cpp(Y, r, n, q);

	arma::mat xy_mm = mx * my;
	arma::mat xy_ele = mx % my; // element-wise multiplication

	if (estimation_method == "u")
	{
		arma::vec diag_mx = mx.diag();
		arma::vec diag_my = my.diag();

		arma::mat mmx = diag_mx * arma::rowvec(n - 1, arma::fill::ones);
		mmx = mmx % my;

		arma::mat mmy = diag_my * arma::rowvec(n - 1, arma::fill::ones);
		mmy = mmy % mx;

		double a = arma::accu(xy_mm.diag());
		double b = arma::accu(diag_mx % diag_my);

		double sr1 = a - b;
		double sr2 =
			arma::accu(xy_mm) - a + 2 * b - arma::accu(mmx) - arma::accu(mmy);
		double sr3 = (arma::accu(mx) - arma::accu(diag_mx)) *
						 (arma::accu(my) - arma::accu(diag_my)) -
					 2 * sr1 - 4 * sr2;

		out[0] = sr3 / ((n - 1.0) * (n - 2) * (n - 3) * (n - 4));
		out[1] = sr1 / ((n - 1.0) * (n - 2)) -
				 2 * sr2 / ((n - 1.0) * (n - 2) * (n - 3)) +
				 sr3 / ((n - 1.0) * (n - 2) * (n - 3) * (n - 4));
	}
	else if (estimation_method == "v")
	{
		double sr = arma::accu(xy_mm) / pow(n, 3);

		out[0] = arma::accu(mx) * arma::accu(my) / pow(n, 4);
		out[1] = arma::accu(xy_ele) / pow(n, 2) + out[0] - 2 * sr;
	}
	else
	{
		stop("The parameter \"estimation.method\" should be \"u\" or \"v\".");
	}

	return out;
}

arma::vec pcov_va_cpp(const arma::mat &X, const arma::mat &Y,
					  const std::string &estimation_method,
					  const int n_threads = 4)
{
	int n = X.n_rows;
	int p = X.n_cols;
	int q = Y.n_cols;

	arma::mat results(2, n, fill::zeros); // pcov results

	if (n_threads == 1)
	{
		for (int i = 0; i < n; i++)
		{
			results.col(i) = one_pcov_cpp(i, X, Y, n, p, q, estimation_method);
		}
	}
	else if (n_threads > 1)
	{
		omp_set_num_threads(n_threads);
#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			results.col(i) = one_pcov_cpp(i, X, Y, n, p, q, estimation_method);
		}
	}

	// pcov results by row
	return mean(results, 1);
}


// [[Rcpp::export]]
double pcov_cpp(const arma::mat &X, const arma::mat &Y,
				const std::string &estimation_method, const int n_threads = 4)
{
	if (X.n_rows != Y.n_rows)
	{
		Rcpp::stop("The numbers of rows in two matrices should be equal.");
	}
	if (X.n_rows <= 3)
	{
		Rcpp::stop("The number of rows in X should be larger than 3.");
	}
	arma::vec t = pcov_va_cpp(X, Y, estimation_method, n_threads);
	return t(1);
}


// [[Rcpp::export]]
double pcor_cpp(const arma::mat &X, const arma::mat &Y,
				const std::string &estimation_method = "u",
				const int n_threads = 4)
{
	double t1 = pcov_cpp(X, Y, estimation_method, n_threads);
	double t2 = pcov_cpp(X, X, estimation_method, n_threads);
	double t3 = pcov_cpp(Y, Y, estimation_method, n_threads);

	return t1 / std::sqrt(t2) / std::sqrt(t3);
}

// [[Rcpp::export]]
Rcpp::List pcov_test_cpp(const arma::mat &X, const arma::mat &Y,
						 const std::string &estimation_method = "u",
						 const int times = 199,
						 const int n_threads = 4)
{
	double value = pcov_cpp(X, Y, estimation_method, n_threads);
	arma::vec values(times);
	omp_set_num_threads(n_threads);
	int n = X.n_rows;
#pragma omp parallel for
	for (int t = 0; t < times; ++t)
	{
		values(t) = pcov_cpp(X.rows(arma::randperm(n)), Y, estimation_method, 1);
	}

	double count = arma::accu(values < value);
	double p_value = 1.0 - (count / static_cast<double>(times));

	return Rcpp::List::create(
		Rcpp::Named("method") =
			"Projection Covariance Permutation Test of Independence",
		Rcpp::Named("stat.value") = value, Rcpp::Named("p.value") = p_value);
}

double chisq_va_cpp(const arma::mat &X, const arma::mat &Y,
					const std::string &estimation_method,
					const int n_threads = 4)
{
	if (X.n_rows != Y.n_rows)
	{
		Rcpp::stop("The numbers of rows in two matrices should be equal.");
	}
	if (X.n_rows <= 3)
	{
		Rcpp::stop("The number of rows in X should be larger than 3.");
	}

	arma::vec t = pcov_va_cpp(X, Y, estimation_method, n_threads);
	return t(1) / (M_PI * M_PI - t(0));
}

// [[Rcpp::export]]
Rcpp::List pcor_test_cpp(const arma::mat &X, const arma::mat &Y,
						 const std::string &estimation_method = "u",
						 const int times = 199, const int n_threads = 4)
{

	int n = X.n_rows;

	double value = chisq_va_cpp(X, Y, estimation_method, n_threads);
	double stat_value = n * value;

	arma::vec pcor_permu(times);

	omp_set_num_threads(n_threads);
#pragma omp parallel for
	for (int t = 0; t < times; ++t)
	{
		pcor_permu(t) =
			chisq_va_cpp(X.rows(arma::randperm(n)), Y, estimation_method, 1);
	}

	double count = arma::accu(pcor_permu < value);
	double p_value = 1.0 - (count / static_cast<double>(times));

	return Rcpp::List::create(
		Rcpp::Named("method") =
			"Projection Correlation Permutation Test of Independence",
		Rcpp::Named("stat.value") = stat_value, Rcpp::Named("p.value") = p_value);
}
