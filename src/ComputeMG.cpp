
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeProlongation_ref.hpp"
#include "ComputeRestriction_ref.hpp"

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
	if(A.optimizationData == 0 || r.optimizationData == 0 || x.optimizationData == 0){
		A.isMgOptimized = false;
		return ComputeMG_ref(A, r, x);
	}

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;
    ierr = ComputeSPMV(A, x, *A.mgData->Axf); if (ierr!=0) return ierr;
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction(A, r);  if (ierr!=0) return ierr;
    ierr = ComputeMG(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;
    ierr = ComputeProlongation(A, x);  if (ierr!=0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;
  }
  else {
    ierr = ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;
  }
  return 0;
}

class RestrictionFunctor{
	public:
		double_1d_type Axfv;
		double_1d_type rfv;
		double_1d_type rcv;
		local_int_1d_type f2c;

		RestrictionFunctor(double_1d_type &Axfv_, double_1d_type &rfv_, double_1d_type &rcv_, local_int_1d_type &f2c_):
			Axfv(Axfv_), rfv(rfv_), rcv(rcv_), f2c(f2c_){}

		KOKKOS_INLINE_FUNCTION
		void operator()(const int &i) const{
			rcv(i) = rfv(f2c(i)) - Axfv(f2c(i));
		}
};

class ProlongationFunctor{
	public:
		double_1d_type xfv;
		double_1d_type xcv;
		local_int_1d_type f2c;

		ProlongationFunctor(double_1d_type &xfv_, double_1d_type &xcv_, local_int_1d_type f2c_):
			xfv(xfv_), xcv(xcv_), f2c(f2c_){}

		KOKKOS_INLINE_FUNCTION
		void operator()(const int &i) const{
			xfv(f2c(i)) += xcv(i);
		}
};

int ComputeRestriction(const SparseMatrix & A, const Vector & rf){
	Optimatrix * A_Optimized = (Optimatrix *) A.optimizationData;
	MGData A_MG = *A.mgData;
	Optivector * Axf_Optimized = (Optivector *) A_MG.Axf->optimizationData;
	Optivector * rcv_Optimized = (Optivector *) A_MG.rc->optimizationData;
	Optivector * rfv_Optimized = (Optivector *) rf.optimizationData;
	local_int_1d_type f2c = A_Optimized->f2cOperator;
	double_1d_type Axfv = Axf_Optimized->values;
	double_1d_type rcv = rcv_Optimized->values;
	double_1d_type rfv = rfv_Optimized->values;
	local_int_t nc = A.mgData->rc->localLength;

	Kokkos::parallel_for(nc, RestrictionFunctor(Axfv, rfv, rcv, f2c));
	return 0;
}

int ComputeProlongation(const SparseMatrix &Af, Vector & xf){
	Optimatrix * Af_Optimized = (Optimatrix *) Af.optimizationData;
	MGData Af_MG = *Af.mgData;
	Optivector * xcv_Optimized = (Optivector *) Af_MG.xc->optimizationData;
	Optivector * xfv_Optimized = (Optivector *) xf.optimizationData;
	local_int_1d_type f2c = Af_Optimized->f2cOperator;
	double_1d_type xcv = xcv_Optimized->values;
	double_1d_type xfv = xfv_Optimized->values;
	local_int_t nc = Af.mgData->rc->localLength;

	Kokkos::parallel_for(nc, ProlongationFunctor(xfv, xcv, f2c));
	return 0;
}
