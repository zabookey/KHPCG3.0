#ifdef SYMGS_LEVEL
#include "LevelSYMGS.hpp"

#ifdef KOKKOS_TEAM
typedef Kokkos::TeamPolicy<>              team_policy ;
typedef team_policy::member_type team_member ;
class LeveledSweep{
public:
  local_matrix_type A;

  local_int_t level_set_begin;
  local_int_t level_set_end;

  local_int_1d_type lev_ind;

  double_1d_type rv, xv;

  LeveledSweep(const local_int_t level_set_begin_, const local_int_t level_set_end_,
    const local_matrix_type & A_, const local_int_1d_type & lev_ind_,
    const double_1d_type &rv_, double_1d_type & xv_):
    level_set_begin(level_set_begin_), level_set_end(level_set_end_), A(A_),
    lev_ind(lev_ind_), rv(rv_), xv(xv_){}

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member & teamMember) const{
    int ii = teamMember.league_rank() * teamMember.team_size() + teamMember.team_rank() + level_set_begin;
    if(ii >= level_set_end) return;
    int crow = lev_ind(ii);
    int row_begin = A.graph.row_map(crow);
    int row_end = A.graph.row_map(crow+1);

    bool am_i_the_diagonal = false;
    double diagonal = 1;
    double sum = 0;
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember, row_end - row_begin),
      [&](int i, double & valueToUpdate){
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
class LeveledSweep{
public:
  local_int_t levels_row;
  local_int_1d_type lev_ind;

  local_matrix_type A;
  double_1d_type rv, xv;
  local_int_1d_type matrixDiagonal;

  LeveledSweep(const local_int_t levels_row_, const local_int_1d_type & lev_ind_,
    const local_matrix_type & A_, const double_1d_type & rv_, double_1d_type & xv_,
    const local_int_1d_type matrixDiagonal_):
    levels_row(levels_row_), lev_ind(lev_ind_), A(A_), rv(rv_), xv(xv_), matrixDiagonal(matrixDiagonal_){}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & i) const{
    local_int_t currentRow = lev_ind(levels_row+i);
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



int LevelSYMGS( const SparseMatrix & A, const Vector & r, Vector & x){
assert(x.localLength == A.localNumberOfColumns); // Make sure x contains space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif
  Optimatrix * A_Optimized = (Optimatrix*) A.optimizationData;
  local_matrix_type localMatrix = A_Optimized->localMatrix;
  local_int_1d_type matrixDiagonal = A_Optimized->matrixDiagonal;
  LevelScheduler * levels = A_Optimized->levels;
  const int f_numLevels = levels->f_numberOfLevels;
  const int b_numLevels = levels->b_numberOfLevels;
  host_local_int_1d_type f_lev_map = levels->f_lev_map;
  local_int_1d_type f_lev_ind = levels->f_lev_ind;
  host_local_int_1d_type b_lev_map = levels->b_lev_map;
  local_int_1d_type b_lev_ind = levels->b_lev_ind;
  local_int_1d_type f_row_level = levels->f_row_level;
  local_int_1d_type b_row_level = levels->b_row_level;

  Optivector * r_Optimized = (Optivector *) r.optimizationData;
  double_1d_type r_values = r_Optimized->values;

  Optivector * x_Optimized = (Optivector *) x.optimizationData;
  double_1d_type x_values = x_Optimized->values;

  double_1d_type z("z", x_values.dimension_0());
#ifdef KOKKOS_TEAM
  const int row_per_team=256;
  const int vector_size = 32;
  const int teamSizeMax = 8;
  for(int i = 0; i < f_numLevels; i++){
    int level_index_begin = f_lev_map(i);
    int level_index_end = f_lev_map(i+1);
    int numberOfTeams = level_index_end - level_index_begin;
    Kokkos::parallel_for(team_policy(numberOfTeams/teamSizeMax + 1, teamSizeMax, vector_size),
      LeveledSweep(level_index_begin, level_index_end, localMatrix, f_lev_ind, r_values, x_values));
    execution_space::fence();
  }
  for(int i = 0; i < b_numLevels; i++){
    int level_index_begin = b_lev_map(i);
    int level_index_end = b_lev_map(i+1);
    int numberOfTeams = level_index_end - level_index_begin;
    Kokkos::parallel_for(team_policy(numberOfTeams/teamSizeMax + 1, teamSizeMax, vector_size),
      LeveledSweep(level_index_begin, level_index_end, localMatrix, b_lev_ind, r_values, x_values));
    execution_space::fence();
  }

  
#else
  for(int i = 0; i < f_numLevels; i++){
    int start = f_lev_map(i);
    int end = f_lev_map(i+1);
    Kokkos::parallel_for(end - start, LeveledSweep(start, f_lev_ind, localMatrix, r_values, x_values, matrixDiagonal));
  }
  for(int i = 0; i < b_numLevels; i++){
    int start = b_lev_map(i);
    int end = b_lev_map(i+1);
    Kokkos::parallel_for(end - start, LeveledSweep(start, b_lev_ind, localMatrix, r_values, x_values, matrixDiagonal));
  }
#endif

  return (0);
}
#endif
