
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
 @file Vector.hpp

 HPCG data structures for dense vectors
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <cassert>
#include <cstdlib>
#include "Geometry.hpp"

#include "KokkosSetup.hpp"

struct Vector_STRUCT {
  local_int_t localLength;  //!< length of local portion of the vector
  double * values;          //!< array of values
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;
	double_1d_type optimizedValues;

};
typedef struct Vector_STRUCT Vector;

struct Optivector_STRUCT{
  double_1d_type values;
};
typedef struct Optivector_STRUCT Optivector;

/*!
  Initializes input vector.

  @param[in] v
  @param[in] localLength Length of local portion of input vector
 */
inline void InitializeVector(Vector & v, local_int_t localLength) {
  v.localLength = localLength;
  v.values = new double[localLength];
  v.optimizationData = 0;
  return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
inline void ZeroVector(Vector & v) {
  local_int_t localLength = v.localLength;
  double * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = 0.0;
	//TODO Check if this is legal
	if(v.optimizationData != 0){
		Optivector * v_Optimized = (Optivector *) v.optimizationData;
		Kokkos::deep_copy(v_Optimized->values, 0.0);
	}
  return;
}
/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */
inline void ScaleVectorValue(Vector & v, local_int_t index, double value) {
  assert(index>=0 && index < v.localLength);
  double * vv = v.values;
  vv[index] *= value;
	if(v.optimizationData != 0){
		Optivector * v_Optimized = (Optivector *) v.optimizationData;
		double_1d_type v_values = v_Optimized->values;
		host_double_1d_type host_v_values = Kokkos::create_mirror_view(v_values);
		host_v_values(index) *= value;
		Kokkos::deep_copy(v_values, host_v_values);
	}
  return;
}
/*!
  Fill the input vector with pseudo-random values.

  @param[in] v
 */
inline void FillRandomVector(Vector & v) {
  local_int_t localLength = v.localLength;
  double * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = rand() / (double)(RAND_MAX) + 1.0;
  return;
}
/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
inline void CopyVector(const Vector & v, Vector & w) {
  local_int_t localLength = v.localLength;
  assert(w.localLength >= localLength);
  double * vv = v.values;
  double * wv = w.values;
  for (int i=0; i<localLength; ++i) wv[i] = vv[i];
	if(v.optimizationData != 0 &&  w.optimizationData != 0){
		std::cout<<"OPTIMIZED COPY"<<std::endl;
		Optivector * v_Optimized = (Optivector *) v.optimizationData;
		double_1d_type v_Values = v_Optimized->values;
		host_double_1d_type host_v_Values = Kokkos::create_mirror_view(v_Values);
		Kokkos::deep_copy(host_v_Values, v_Values);
		Optivector * w_Optimized = (Optivector *) w.optimizationData;
		double_1d_type w_Values = w_Optimized->values;
		host_double_1d_type host_w_Values = Kokkos::create_mirror_view(w_Values);
		Kokkos::deep_copy(host_w_Values, w_Values);
		for(int i = 0; i < localLength; ++i)
			host_w_Values(i) = host_v_Values(i);
		Kokkos::deep_copy(w_Values, host_w_Values);
	}
  return;
}


/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteVector(Vector & v) {

  delete [] v.values;
  v.localLength = 0;
  if(v.optimizationData != 0){delete (Optivector*) v.optimizationData; v.optimizationData = 0;}
  return;
}

#endif // VECTOR_HPP
