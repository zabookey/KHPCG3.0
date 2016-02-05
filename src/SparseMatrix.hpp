
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
 @file SparseMatrix.hpp

 HPCG data structures for the sparse matrix
 */

#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <map>
#include <vector>
#include <cassert>
#include "Geometry.hpp"
#include "Vector.hpp"
#include "MGData.hpp"

#include "KokkosSetup.hpp"

struct SparseMatrix_STRUCT {
  char  * title; //!< name of the sparse matrix
  Geometry * geom; //!< geometry associated with this matrix
  global_int_t totalNumberOfRows; //!< total number of matrix rows across all processes
  global_int_t totalNumberOfNonzeros; //!< total number of matrix nonzeros across all processes
  local_int_t localNumberOfRows; //!< number of rows local to this process
  local_int_t localNumberOfColumns;  //!< number of columns local to this process
  local_int_t localNumberOfNonzeros;  //!< number of nonzeros local to this process
  char  * nonzerosInRow;  //!< The number of nonzeros in a row will always be 27 or fewer
  global_int_t ** mtxIndG; //!< matrix indices as global values
  local_int_t ** mtxIndL; //!< matrix indices as local values
  double ** matrixValues; //!< values of matrix entries
  double ** matrixDiagonal; //!< values of matrix diagonal entries
  std::map< global_int_t, local_int_t > globalToLocalMap; //!< global-to-local mapping
  std::vector< global_int_t > localToGlobalMap; //!< local-to-global mapping
  mutable bool isDotProductOptimized;
  mutable bool isSpmvOptimized;
  mutable bool isMgOptimized;
  mutable bool isWaxpbyOptimized;
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  mutable struct SparseMatrix_STRUCT * Ac; // Coarse grid matrix
  mutable MGData * mgData; // Pointer to the coarse level data for this fine matrix
  void * optimizationData;  // pointer that can be used to store implementation-specific data

#ifndef HPCG_NO_MPI
  local_int_t numberOfExternalValues; //!< number of entries that are external to this process
  int numberOfSendNeighbors; //!< number of neighboring processes that will be send local data
  local_int_t totalToBeSent; //!< total number of entries to be sent
  local_int_t * elementsToSend; //!< elements to send to neighboring processes
  int * neighbors; //!< neighboring processes
  local_int_t * receiveLength; //!< lenghts of messages received from neighboring processes
  local_int_t * sendLength; //!< lenghts of messages sent to neighboring processes
  double * sendBuffer; //!< send buffer for non-blocking sends
#endif
};
typedef struct SparseMatrix_STRUCT SparseMatrix;

struct Optimatrix_STRUCT{
  local_matrix_type localMatrix;
  global_matrix_type globalMatrix;
  local_int_1d_type matrixDiagonal; // values(matrixDiagonal(i)) will return value on diagonal of row i.
  local_int_1d_type f2cOperator; // Use this instead of the one in MGData so it can be used in Kokkos kernels.
};
typedef struct Optimatrix_STRUCT Optimatrix;
/*!
  Initializes the known system matrix data structure members to 0.

  @param[in] A the known system matrix
 */
inline void InitializeSparseMatrix(SparseMatrix & A, Geometry * geom) {
  A.title = 0;
  A.geom = geom;
  A.totalNumberOfRows = 0;
  A.totalNumberOfNonzeros = 0;
  A.localNumberOfRows = 0;
  A.localNumberOfColumns = 0;
  A.localNumberOfNonzeros = 0;
  A.nonzerosInRow = 0;
  A.mtxIndG = 0;
  A.mtxIndL = 0;
  A.matrixValues = 0;
  A.matrixDiagonal = 0;

  // Optimization is ON by default. The code that switches it OFF is in the
  // functions that are meant to be optimized.
  A.isDotProductOptimized = true;
  A.isSpmvOptimized       = true;
  A.isMgOptimized      = true;
  A.isWaxpbyOptimized     = true;

#ifndef HPCG_NO_MPI
  A.numberOfExternalValues = 0;
  A.numberOfSendNeighbors = 0;
  A.totalToBeSent = 0;
  A.elementsToSend = 0;
  A.neighbors = 0;
  A.receiveLength = 0;
  A.sendLength = 0;
  A.sendBuffer = 0;
#endif
  A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
  A.Ac =0;
  return;
}

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before call to this function).
 */
inline void CopyMatrixDiagonal(SparseMatrix & A, Vector & diagonal) {
    double ** curDiagA = A.matrixDiagonal;
    double * dv = diagonal.values;
    assert(A.localNumberOfRows==diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) dv[i] = *(curDiagA[i]);
		if(A.optimizationData != 0){
			Optimatrix * A_Optimized = (Optimatrix *) A.optimizationData;
			local_matrix_type A_localMatrix = A_Optimized->localMatrix;
			host_values_type host_A_Values = Kokkos::create_mirror_view(A_localMatrix.values);
			local_int_1d_type A_diagonal_entries = A_Optimized->matrixDiagonal;
			host_local_int_1d_type host_A_diag_entries = Kokkos::create_mirror_view(A_diagonal_entries);
			Kokkos::deep_copy(host_A_Values, A_localMatrix.values);
			Kokkos::deep_copy(host_A_diag_entries, A_diagonal_entries);
			if(diagonal.optimizationData != 0){
				Optivector * diagonal_Optimized = (Optivector *) diagonal.optimizationData;
				double_1d_type d_values = diagonal_Optimized->values;
				host_double_1d_type host_d_values = Kokkos::create_mirror_view(d_values);
				Kokkos::deep_copy(host_d_values, d_values);
				for(local_int_t i = 0; i < A.localNumberOfRows; ++i)
					host_d_values(i) = host_A_Values(host_A_diag_entries(i));
				Kokkos::deep_copy(d_values, host_d_values);
			}
			for(local_int_t i = 0; i < A.localNumberOfRows; ++i)
				dv[i] = host_A_Values(host_A_diag_entries(i));
		}
  return;
}
/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing matrix diagonal values.
 */
inline void ReplaceMatrixDiagonal(SparseMatrix & A, Vector & diagonal) {
    double ** curDiagA = A.matrixDiagonal;
    double * dv = diagonal.values;
    assert(A.localNumberOfRows==diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) *(curDiagA[i]) = dv[i];
		if(A.optimizationData != 0){
			Optimatrix * A_Optimized = (Optimatrix *) A.optimizationData;
			local_matrix_type A_localMatrix = A_Optimized->localMatrix;
			host_values_type host_A_values = Kokkos::create_mirror_view(A_localMatrix.values);
			local_int_1d_type A_diagonal_entries = A_Optimized->matrixDiagonal;
			host_local_int_1d_type host_A_diag_entries = Kokkos::create_mirror_view(A_diagonal_entries);
			Kokkos::deep_copy(host_A_values, A_localMatrix.values);
			Kokkos::deep_copy(host_A_diag_entries, A_diagonal_entries);
			if(diagonal.optimizationData != 0){
				Optivector * diagonal_Optimized = (Optivector *) diagonal.optimizationData;
				double_1d_type d_values = diagonal_Optimized->values;
				host_double_1d_type host_d_values = Kokkos::create_mirror_view(d_values);
				Kokkos::deep_copy(host_d_values, d_values);
				for(local_int_t i = 0; i < A.localNumberOfRows; ++i)
					host_A_values(host_A_diag_entries(i)) = host_d_values(i);
			} else {
				for(local_int_t i = 0; i < A.localNumberOfRows; ++i)
					host_A_values(host_A_diag_entries(i)) = dv[i];
			}
			Kokkos::deep_copy(A_localMatrix.values, host_A_values);
		}
  return;
}
/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteMatrix(SparseMatrix & A) {

  for (local_int_t i = 0; i< A.localNumberOfRows; ++i) {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndG[i];
    delete [] A.mtxIndL[i];
  }

  if (A.title)                  delete [] A.title;
  if (A.nonzerosInRow)             delete [] A.nonzerosInRow;
  if (A.mtxIndG) delete [] A.mtxIndG;
  if (A.mtxIndL) delete [] A.mtxIndL;
  if (A.matrixValues) delete [] A.matrixValues;
  if (A.matrixDiagonal)           delete [] A.matrixDiagonal;

#ifndef HPCG_NO_MPI
  if (A.elementsToSend)       delete [] A.elementsToSend;
  if (A.neighbors)              delete [] A.neighbors;
  if (A.receiveLength)            delete [] A.receiveLength;
  if (A.sendLength)            delete [] A.sendLength;
  if (A.sendBuffer)            delete [] A.sendBuffer;
#endif

  if (A.geom!=0) { delete A.geom; A.geom = 0;}
  if (A.optimizationData != 0) {delete (Optimatrix*) A.optimizationData; A.optimizationData = 0;}
  if (A.Ac!=0) { DeleteMatrix(*A.Ac); delete A.Ac; A.Ac = 0;} // Delete coarse matrix
  if (A.mgData!=0) { DeleteMGData(*A.mgData); delete A.mgData; A.mgData = 0;} // Delete MG data
  return;
}



#endif // SPARSEMATRIX_HPP
