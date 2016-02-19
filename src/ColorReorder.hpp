#ifndef REORDER_HPP
#define REORDER_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "KokkosSetup.hpp"

int ColorReorder(SparseMatrix & A, Vector & x, Vector & b);

#endif
