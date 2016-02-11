
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

#ifndef COMPUTESYMGS_HPP
#define COMPUTESYMGS_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "KokkosSetup.hpp"

#ifdef SYMGS_COLOR
#include "ColorSYMGS.hpp"
#endif
#ifdef SYMGS_LEVEL
#include "LevelSYMGS.hpp"
#endif

int ComputeSYMGS( const SparseMatrix  & A, const Vector & r, Vector & x);

#endif // COMPUTESYMGS_HPP
