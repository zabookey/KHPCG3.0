
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
  	OptimizeCGData(data);
	OptimizeVector(b);
	OptimizeVector(x);
	OptimizeVector(xexact);

  #ifdef SYMGS_COLOR
    doColoring(A);
  #ifdef REORDER
    ColorReorder(A,x,b);
    SparseMatrix * curMatrix = &A;
    while(curMatrix->Ac != 0){
    	ColorReorder(*curMatrix->Ac, *curMatrix->mgData->rc, *curMatrix->mgData->xc);
    	curMatrix = curMatrix->Ac;
    }

  #endif
  #endif
  #ifdef SYMGS_LEVEL
    levelSchedule(A);
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
  local_int_1d_type matrixDiagonal("Matrix Diagonal", A.localNumberOfRows);
  host_local_int_1d_type host_matrixDiagonal = Kokkos::create_mirror_view(matrixDiagonal);
	local_int_1d_type f2cOperator("f2cOperator", A.localNumberOfRows);
	host_local_int_1d_type host_f2cOperator = Kokkos::create_mirror_view(f2cOperator);
	local_int_t index = 0;
	host_rowMap(0) = 0;
//TODO Make this parallel so I don't need to use mirrors and copies
	for(int i = 0; i < A.localNumberOfRows; i++){
		for(int j = 0; j < A.nonzerosInRow[i]; j++){
			host_values(index) = A.matrixValues[i][j];
			host_lIndexMap(index) = A.mtxIndL[i][j];
			if(host_lIndexMap(index) == i)
				host_matrixDiagonal(i) = index;
			host_gIndexMap(index) = A.mtxIndG[i][j];
			index++;
		}
		//host_rowMap(i+1) = host_rowMap(i) + A.nonzerosInRow[i];
		host_rowMap(i+1) = index;
	}
	Kokkos::deep_copy(values, host_values);
	Kokkos::deep_copy(gIndexMap, host_gIndexMap);
	Kokkos::deep_copy(lIndexMap, host_lIndexMap);
	Kokkos::deep_copy(rowMap, host_rowMap);
	Kokkos::deep_copy(matrixDiagonal, host_matrixDiagonal);
	global_matrix_type globalMatrix = global_matrix_type("Matrix: Global", A.localNumberOfRows, A.localNumberOfRows, A.localNumberOfNonzeros, values, rowMap, gIndexMap);
	local_matrix_type localMatrix = local_matrix_type("Matrix: Local", A.localNumberOfRows, A.localNumberOfRows, A.localNumberOfNonzeros, values, rowMap, lIndexMap);
	//Create the optimatrix structure and assign it to A
	Optimatrix* optimized = new Optimatrix;
	optimized->localMatrix = localMatrix;
	optimized->globalMatrix = globalMatrix;
	optimized->matrixDiagonal = matrixDiagonal;
	A.optimizationData = optimized;
	if(A.Ac!=0){
		local_int_t * f2c = A.mgData->f2cOperator;
		for(int i = 0; i < A.localNumberOfRows; i++){
			host_f2cOperator(i) = f2c[i];
		}
		Kokkos::deep_copy(f2cOperator, host_f2cOperator);
		optimized->f2cOperator = f2cOperator;
		OptimizeMGData(*A.mgData);
		OptimizeMatrix(*A.Ac);
	}
}

void OptimizeVector(Vector & v){
	double_1d_type values = double_1d_type("Vector: Values", v.localLength);
	host_double_1d_type host_values = Kokkos::create_mirror_view(values);
	for(int i = 0; i < v.localLength; i++){
		host_values(i) = v.values[i];
	}
	Kokkos::deep_copy(values, host_values);
	//Create the optivector structure and assign it to v
	Optivector* optimized = new Optivector;
	optimized->values = values;
	v.optimizationData = optimized;
}

void OptimizeCGData(CGData & data){
  OptimizeVector(data.r);
  OptimizeVector(data.z);
  OptimizeVector(data.p);
  OptimizeVector(data.Ap);
}

void OptimizeMGData(MGData & data){
  OptimizeVector(*data.rc);
  OptimizeVector(*data.xc);
  OptimizeVector(*data.Axf);
}
