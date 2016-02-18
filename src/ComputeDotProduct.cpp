
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"
#include <cassert>

  class Dotproduct {

    private:
    const_double_1d_type  xv;
    const_double_1d_type  yv;

    public:
    typedef double value_type;
    Dotproduct(const double_1d_type  & xValues,const double_1d_type  & yValues){
      xv = xValues;
      yv = yValues;
    }
    KOKKOS_INLINE_FUNCTION
    void operator()(local_int_t i, double &final)const{
    final += xv(i) * yv(i);
    }

  };

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {
  
    if(x.optimizationData == 0 || y.optimizationData == 0){
      isOptimized = false;
      return(ComputeDotProduct_ref(n,x,y,result,time_allreduce));
    }

  // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
  assert(x.localLength >= n);
  assert(y.localLength >= n);

  Optivector * x_optimized = (Optivector*) x.optimizationData;
  double_1d_type x_values = x_optimized->values;
  Optivector * y_optimized = (Optivector*) y.optimizationData;
  double_1d_type y_values = y_optimized->values;

  double local_result = 0.0;
  Kokkos::parallel_reduce(n,Dotproduct(x_values, y_values), local_result);

  #ifndef HPCG_NO_MPI
    double t0  = mytimer();
    double global_result = 0.0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    result = global_result;
    time_allreduce += mytimer() - t0;
  #else
    result = local_result;
  #endif

    return(0);
}
