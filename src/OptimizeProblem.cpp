
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"
#include <iostream>
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

  // This function can be used to completely transform any part of the data structures.
  // Right now it does nothing, so compiling with a check for unused variables results in complaints

	OptimizeMatrix(A);
	OptimizeVector(b);
	OptimizeVector(x);
	OptimizeVector(xexact);

#if defined(HPCG_USE_MULTICOLORING)
  const local_int_t nrow = A.localNumberOfRows;
  std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
  int totalColors = 1;
  colors[0] = 0; // first point gets color 0

  // Finds colors in a greedy (a likely non-optimal) fashion.

  for (local_int_t i=1; i < nrow; ++i) {
    if (colors[i] == nrow) { // if color not assigned
      std::vector<int> assigned(totalColors, 0);
      int currentlyAssigned = 0;
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      for (int j=0; j< currentNumberOfNonzeros; j++) { // scan neighbors
        local_int_t curCol = currentColIndices[j];
        if (curCol < i) { // if this point has an assigned color (points beyond `i' are unassigned)
          if (assigned[colors[curCol]] == 0)
            currentlyAssigned += 1;
          assigned[colors[curCol]] = 1; // this color has been used before by `curCol' point
        } // else // could take advantage of indices being sorted
      }

      if (currentlyAssigned < totalColors) { // if there is at least one color left to use
        for (int j=0; j < totalColors; ++j)  // try all current colors
          if (assigned[j] == 0) { // if no neighbor with this color
            colors[i] = j;
            break;
          }
      } else {
        if (colors[i] == nrow) {
          colors[i] = totalColors;
          totalColors += 1;
        }
      }
    }
  }

  std::vector<local_int_t> counters(totalColors);
  for (local_int_t i=0; i<nrow; ++i)
    counters[colors[i]]++;

  local_int_t old, old0;
  for (int i=1; i < totalColors; ++i) {
    old0 = counters[i];
    counters[i] = counters[i-1] + old;
    old = old0;
  }
  counters[0] = 0;

  // translate `colors' into a permutation
  for (local_int_t i=0; i<nrow; ++i) // for each color `c'
    colors[i] = counters[colors[i]]++;
#endif

  return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  return 0.0;

}

void OptimizeMatrix(SparseMatrix & A){
	values_type values("CrsMatrix: Values", A.localNumberOfNonzeros);
	host_values_type host_values = Kokkos::create_mirror_view(values);
	global_index_type gIndexMap("CrsMatrix: GlobalIndexMap", A.localNumberOfNonzeros);
	host_global_index_type host_gIndexMap = Kokkos::create_mirror_view(gIndexMap);
	local_index_type lIndexMap("CrsMatrix: LocalIndexMap", A.localNumberOfNonzeros);
	host_local_index_type host_lIndexMap = Kokkos::create_mirror_view(lIndexMap);
	non_const_row_map_type rowMap("CrsMatrix: RowMap", A.localNumberOfRows+1);
	host_non_const_row_map_type host_rowMap = Kokkos::create_mirror_view(rowMap);
	local_int_t index = 0;
	host_rowMap(0) = 0;
//TODO Make this parallel so I don't need to use mirrors and copies
	for(int i = 0; i < A.localNumberOfRows; i++){
		for(int j = 0; j < A.nonzerosInRow[i]; j++){
			host_values(index) = A.matrixValues[i][j];
			host_lIndexMap(index) = A.mtxIndL[i][j];
			host_gIndexMap(index) = A.mtxIndG[i][j];
			index++;
		}
		host_rowMap(i+1) = host_rowMap(i) + A.nonzerosInRow[i];
	}
	Kokkos::deep_copy(values, host_values);
	Kokkos::deep_copy(gIndexMap, host_gIndexMap);
	Kokkos::deep_copy(lIndexMap, host_lIndexMap);
	Kokkos::deep_copy(rowMap, host_rowMap);
	global_matrix_type globalMatrix = global_matrix_type("Matrix: Global", A.localNumberOfRows, A.localNumberOfRows, A.localNumberOfNonzeros, values, rowMap, gIndexMap);
	local_matrix_type localMatrix = local_matrix_type("Matrix: Local", A.localNumberOfRows, A.localNumberOfRows, A.localNumberOfNonzeros, values, rowMap, lIndexMap);
	//Create the optimatrix structure and assign it to A
	Optimatrix optimized;
	optimized.localMatrix = localMatrix;
	optimized.globalMatrix = globalMatrix;
	A.optimizationData = &optimized;
}

void OptimizeVector(Vector & v){
	double_1d_type values = double_1d_type("Vector: Values", v.localLength);
	host_double_1d_type host_values = Kokkos::create_mirror_view(values);
	for(int i = 0; i < v.localLength; i++){
		host_values(i) = v.values[i];
	}
	Kokkos::deep_copy(values, host_values);
	//Create the optivector structure and assign it to v
	Optivector optimized;
	optimized.values = values;
	v.optimizationData = &optimized;
}
