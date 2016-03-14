#ifdef SYMGS_COLOR
#include "ColorSYMGS.hpp"

#ifdef KOKKOS_TEAM
typedef Kokkos::TeamPolicy<> team_policy;
typedef typename team_policy::member_type team_member;
class ColouredSweep{
public:
  local_matrix_type A;

  local_int_t color_set_begin;
  local_int_t color_set_end;

  local_int_1d_type colors_ind;

  double_1d_type rv, xv;

  ColouredSweep(const local_int_t color_set_begin_, const local_int_t color_set_end_, 
    const local_matrix_type& A_, const local_int_1d_type& colors_ind_, const double_1d_type& rv_, double_1d_type& xv_):
    color_set_begin(color_set_begin_), color_set_end(color_set_end_), A(A_), colors_ind(colors_ind_), rv(rv_), xv(xv_) {}
KOKKOS_INLINE_FUNCTION
  void operator()(const team_member & teamMember) const{
    int ii = teamMember.league_rank() * teamMember.team_size() + teamMember.team_rank() + color_set_begin;
    if(ii >= color_set_end) return;
#ifdef REORDER
		int crow = ii;//colors_ind(ii);
#else
		int crow = colors_ind(ii);
#endif
    int row_begin = A.graph.row_map(crow);
    int row_end = A.graph.row_map(crow+1);

    bool am_i_the_diagonal = false;
    double diagonal = 1;
    double sum = 0;
    Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, row_end - row_begin),
      [&] (int i, double & valueToUpdate) {
        int adjind = i + row_begin;
        int colIndex = A.graph.entries(adjind);
        double val = A.values(adjind);
        if(colIndex == crow){
          diagonal = val;
          am_i_the_diagonal = true;
        }
        else{
          valueToUpdate += val * xv(colIndex);
        }
      }, sum);

    if(am_i_the_diagonal){
      xv(crow) = (rv(crow) - sum)/diagonal;
    }
  }
};
#else
class colouredForwardSweep{
  public:
  local_int_t colors_row;
  local_int_1d_type colors_ind;

  local_matrix_type A;
  double_1d_type rv, xv;
  int_1d_type matrixDiagonal;

  colouredForwardSweep(const local_int_t colors_row_, const local_int_1d_type& colors_ind_,
    const local_matrix_type& A_, const double_1d_type& rv_, double_1d_type& xv_,
    const int_1d_type matrixDiagonal_):
    colors_row(colors_row_), colors_ind(colors_ind_), A(A_), rv(rv_), xv(xv_),
    matrixDiagonal(matrixDiagonal_) {}

KOKKOS_INLINE_FUNCTION
  void operator()(const int & i)const{
#ifdef REORDER
		local_int_t currentRow = colors_row + i;
#else
    local_int_t currentRow = colors_ind(colors_row + i); // This should tell us what row we're doing SYMGS on.
#endif
    int start = A.graph.row_map(currentRow);
    int end = A.graph.row_map(currentRow+1);
    const double currentDiagonal = A.values(matrixDiagonal(currentRow));
    double sum = rv(currentRow);
    for(int j = start; j < end; j++)
      sum -= A.values(j) * xv(A.graph.entries(j));
    sum += xv(currentRow) * currentDiagonal;
    xv(currentRow) = sum/currentDiagonal;
  }
};

class colouredBackSweep{
  public:
  local_int_t colors_row;
  local_int_1d_type colors_ind;
  
  local_matrix_type A;
  double_1d_type rv, xv;
  int_1d_type matrixDiagonal;

  colouredBackSweep(const local_int_t colors_row_, const local_int_1d_type& colors_ind_,
      const local_matrix_type& A_, const double_1d_type& rv_, double_1d_type& xv_,
      const int_1d_type matrixDiagonal_):
      colors_row(colors_row_), colors_ind(colors_ind_), A(A_), rv(rv_), xv(xv_),
      matrixDiagonal(matrixDiagonal_) {}
KOKKOS_INLINE_FUNCTION
  void operator()(const int & i)const{
#ifdef REORDER
		local_int_t currentRow = colors_row + i;
#else
    local_int_t currentRow = colors_ind(colors_row + i); // This should tell us what row we're doing SYMGS on.
#endif
    int start = A.graph.row_map(currentRow);
    int end = A.graph.row_map(currentRow+1);
    const double currentDiagonal = A.values(matrixDiagonal(currentRow));
    double sum = rv(currentRow);
    for(int j = start; j < end; j++)
      sum -= A.values(j) * xv(A.graph.entries(j));
    sum += xv(currentRow) * currentDiagonal;
    xv(currentRow) = sum/currentDiagonal;
  }
};
#endif

int ColorSYMGS( const SparseMatrix & A, const Vector & r, Vector & x){
assert(x.localLength == A.localNumberOfColumns); // Make sure x contains space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif
  Optimatrix* A_Optimized = (Optimatrix*)A.optimizationData;
  local_matrix_type localMatrix = A_Optimized->localMatrix;
  local_int_1d_type matrixDiagonal = A_Optimized->matrixDiagonal;
  local_int_1d_type colors_ind = A_Optimized->colors_ind;
  host_local_int_1d_type host_colors_ind = A_Optimized->host_colors_ind;
  local_int_1d_type colors_map = A_Optimized->colors_map;
  host_local_int_1d_type host_colors_map = A_Optimized->host_colors_map;
  const int numColors = A_Optimized->numColors;

  Optivector * r_Optimized = (Optivector*)r.optimizationData;
  double_1d_type r_values = r_Optimized->values;

  Optivector * x_Optimized = (Optivector*)x.optimizationData;
  double_1d_type x_values = x_Optimized->values;

	 // Forward Sweep!
#ifdef KOKKOS_TEAM
  int vector_size = 32;
  int teamSizeMax = 8;
  for(int i = 0; i < numColors; i++){
    int color_index_begin = host_colors_map(i);
    int color_index_end = host_colors_map(i + 1);
    int numberOfTeams = color_index_end - color_index_begin;
    Kokkos::parallel_for(team_policy(numberOfTeams / teamSizeMax + 1, teamSizeMax, vector_size),
      ColouredSweep(color_index_begin, color_index_end, localMatrix, colors_ind, r_values, x_values));

    execution_space::fence();
  }
  for(int i = numColors - 1; i >= 0; i--){
    int color_index_begin = host_colors_map(i);
    int color_index_end = host_colors_map(i+1);
    int numberOfTeams = color_index_end - color_index_begin;
    Kokkos::parallel_for(team_policy(numberOfTeams / teamSizeMax + 1, teamSizeMax, vector_size),
      ColouredSweep(color_index_begin, color_index_end, localMatrix, colors_ind, r_values, x_values));
    execution_space::fence();
  }
#else
  local_int_t dummy = 0;
  for(int i = 0; i < numColors; i++){
    int start = host_colors_map(i); // Colors start at 1, i starts at 0
    int end = host_colors_map(i+1);
   dummy += end - start;
    Kokkos::parallel_for(end - start, colouredForwardSweep(start, colors_ind, localMatrix, r_values, x_values, matrixDiagonal));
  }
  assert(dummy == A.localNumberOfRows);
 // Back Sweep!
  for(int i = numColors -1; i >= 0; --i){
    int start = host_colors_map(i); // Colors start at 1, i starts at 0
    int end = host_colors_map(i+1);
    Kokkos::parallel_for(end - start, colouredBackSweep(start, colors_ind, localMatrix, r_values, x_values, matrixDiagonal));
  }
#endif
return(0);
}
#endif
