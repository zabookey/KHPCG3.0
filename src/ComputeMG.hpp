
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

#ifndef COMPUTEMG_HPP
#define COMPUTEMG_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "KokkosSetup.hpp"

int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x);

int ComputeRestriction(const SparseMatrix & A, const Vector & rf);

int ComputeProlongation(const SparseMatrix &Af, Vector & xf);
#endif // COMPUTEMG_HPP
