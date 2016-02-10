
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

/*!
  Routine to one step of symmetrix Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in]  A the known system matrix
  @param[in]  x the input vector
  @param[out] y On exit contains the result of one symmetric GS sweep with x as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  if(A.optimizationData == 0 || r.optimizationData == 0 || x.optimizationData == 0){
    return ComputeSYMGS_ref(A, r, x);
  }

  Optimatrix * A_Optimized = (Optimatrix *)A.optimizationData;
  Optivector * r_Optimized = (Optivector *)r.optimizationData;
  Optivector * x_Optimized = (Optivector *)x.optimizationData;
  local_matrix_type localMatrix = A_Optimized->localMatrix;
  local_int_1d_type matrixDiagonal = A_Optimized->matrixDiagonal;
  double_1d_type r_values = r_Optimized->values;
  double_1d_type x_values = x_Optimized->values;
  //Create mirrors since this is run in serial unfortunately...
  const host_values_type host_A_values = Kokkos::create_mirror_view(localMatrix.values);
  const host_local_index_type host_A_entries = Kokkos::create_mirror_view(localMatrix.graph.entries);
  const host_row_map_type host_A_rowMap = Kokkos::create_mirror_view(localMatrix.graph.row_map);
  const host_local_int_1d_type host_matrixDiagonal = Kokkos::create_mirror_view(matrixDiagonal);
  const host_double_1d_type host_r_values = Kokkos::create_mirror_view(r_values);
  host_double_1d_type host_x_values = Kokkos::create_mirror_view(x_values);
  //Now copy all the values to the mirrors
  Kokkos::deep_copy(host_A_values, localMatrix.values);
  Kokkos::deep_copy(host_A_entries, localMatrix.graph.entries);
  Kokkos::deep_copy(host_A_rowMap, localMatrix.graph.row_map);
  Kokkos::deep_copy(host_matrixDiagonal, matrixDiagonal);
  Kokkos::deep_copy(host_r_values, r_values);
  Kokkos::deep_copy(host_x_values, x_values);

  const local_int_t nrow = A.localNumberOfRows;
  //Foreward sweep
  for(local_int_t i = 0; i < nrow; i++){
    local_int_t start = host_A_rowMap(i);
    local_int_t end = host_A_rowMap(i+1);
    const double currentDiagonal = host_A_values(host_matrixDiagonal(i));
    double sum = host_r_values(i);
    for(int j = start; j < end; j++){
      local_int_t curCol = host_A_entries(j);
      sum -= host_A_values(j) * host_x_values(curCol);
    }
    sum += host_x_values(i)*currentDiagonal;
    host_x_values(i) = sum/currentDiagonal;
  }
  //Backsweep
  for(local_int_t i = nrow-1; i >= 0; i--){
    local_int_t start = host_A_rowMap(i);
    local_int_t end = host_A_rowMap(i+1);
    const double currentDiagonal = host_A_values(host_matrixDiagonal(i));
    double sum = host_r_values(i);
    for(int j = start; j < end; j++){
      local_int_t curCol = host_A_entries(j);
      sum -= host_A_values(j) * host_x_values(curCol);
    }
    sum += host_x_values(i)*currentDiagonal;
    host_x_values(i) = sum/currentDiagonal;
  }
  //Copy the updated x data on the host back to the device.
  Kokkos::deep_copy(x_values, host_x_values);
	return(0);
}
