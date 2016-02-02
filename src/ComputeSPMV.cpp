
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  if(A.optimizationData == 0 || x.optimizationData == 0 || y.optimizationData == 0){
    A.isSpmvOptimized = false;
    return ComputeSPMV_ref(A,x,y);
  }
  std::cout<<"SPMV"<<std::endl;
  assert(x.localLength >= A.localNumberOfColumns);
  assert(y.localLength >= A.localNumberOfRows);

  #ifndef HPCG_NOMPI
    ExchangeHalo(A,x);
  #endif

    Optimatrix * A_optimized = (Optimatrix*) A.optimizationData;
    local_matrix_type localMatrix = A_optimized->localMatrix;
    Optivector * x_optimized = (Optivector*) x.optimizationData;
    double_1d_type x_values = x_optimized->values;
    Optivector * y_optimized = (Optivector*) y.optimizationData;
    double_1d_type y_values = y_optimized->values;


    KokkosSparse::spmv("N", 1.0, localMatrix, x_values, 0.0, y_values);
    return(0);
}
