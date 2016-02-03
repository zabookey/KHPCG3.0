
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

 class Waxpby {
		public:
		const_double_1d_type xv;
		const_double_1d_type yv;
		double_1d_type wv;
		double alpha;
		double beta;

		Waxpby(const double_1d_type &xv_,const double_1d_type &yv_, double_1d_type &wv_,const double alpha_,const double beta_):
			xv(xv_), yv(yv_), wv(wv_), alpha(alpha_), beta(beta_)
			{}
		KOKKOS_INLINE_FUNCTION
		void operator() (const int& i)const{
			wv(i) = alpha * xv(i) + beta * yv(i);
		}
	};

	class AlphaOne {
		public:
		const_double_1d_type xv;
		const_double_1d_type yv;
		double_1d_type wv;
		double beta;
	
		AlphaOne(const double_1d_type &xv_, const double_1d_type &yv_, double_1d_type &wv_, const double beta_):
			xv(xv_), yv(yv_), wv(wv_), beta(beta_)
			{}

		KOKKOS_INLINE_FUNCTION
		void operator() (const int& i)const{
			wv(i) = xv(i) + beta * yv(i);
		}
	};

	class BetaOne {
		public:
		const_double_1d_type xv;
		const_double_1d_type yv;
		double_1d_type wv;
		double alpha;
	
		BetaOne(const double_1d_type &xv_, const double_1d_type &yv_, double_1d_type &wv_, const double alpha_):
			xv(xv_), yv(yv_), wv(wv_), alpha(alpha_)
			{}

		KOKKOS_INLINE_FUNCTION
		void operator() (const int& i)const{
			wv(i) = alpha * xv(i) + yv(i);
		}
	};

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  if(x.optimizationData == 0 || y.optimizationData == 0 || w.optimizationData == 0){
    isOptimized = false;
    return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
  }
  assert(x.localLength >= n);
  assert(y.localLength >= n);

  Optivector * x_optimized = (Optivector*) x.optimizationData;
  Optivector * y_optimized = (Optivector*) y.optimizationData;
  Optivector * w_optimized = (Optivector*) w.optimizationData;
  double_1d_type x_values = x_optimized->values;
  double_1d_type y_values = y_optimized->values;
  double_1d_type w_values = w_optimized->values;

  if(alpha == 1.0)
  	Kokkos::parallel_for(n, AlphaOne(x_values, y_values, w_values, beta));
  else if(beta == 1.0)
  	Kokkos::parallel_for(n, BetaOne(x_values, y_values, w_values, alpha));
  else
  	Kokkos::parallel_for(n, Waxpby(x_values, y_values, w_values, alpha, beta));

}
