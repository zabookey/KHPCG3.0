#ifdef SYMGS_LEVEL
#include "LevelScheduler.hpp"

class fillColorsMap{
	public:
	local_int_1d_type colors_map;
	local_int_1d_type colors;

	fillColorsMap(const local_int_1d_type& colors_map_, const local_int_1d_type& colors_):
		colors_map(colors_map_), colors(colors_){}

	// This fills colors_map(i) with the number of indices with color i. Parallel scan will make this the appropriate map.
	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		int color = i+1;
		int total = 0;
		for(unsigned j = 0; j < colors.dimension_0(); j++)
			if(colors(j) == color) total++;
		colors_map(color) = total;
	}
};

class mapScan{
	public:
	local_int_1d_type colors_map;

	mapScan(const local_int_1d_type& colors_map_):
		colors_map(colors_map_){}

// Parallel scan that finishes off setting up colors_map.
	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i, local_int_t & upd, bool final)const{
		upd += colors_map(i);
		if(final)
			colors_map(i) = upd;
	}
};

class fillColorsInd{
	public:
	local_int_1d_type colors_ind;
	local_int_1d_type colors_map;
	local_int_1d_type colors;

	fillColorsInd(local_int_1d_type& colors_ind_, local_int_1d_type& colors_map_,
		local_int_1d_type colors_):
		colors_ind(colors_ind_), colors_map(colors_map_), colors(colors_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		int color = i+1; // Colors start at 1 and i starts at 0.
		int start = colors_map(i);
		for(unsigned j = 0; j < colors.dimension_0(); j++){
			if(colors(j) == color){
				colors_ind(start) = j;
				start++;
			}
		}
		assert(start == colors_map(i+1));//Make sure we only fill up exactly our color. Nothing more nothing less.
	}
};

int levelSchedule(SparseMatrix & A){
	// Number of levels needed for the matrix that owns this
	Optimatrix * A_Optimized = (Optimatrix*) A.optimizationData;
	local_matrix_type localMatrix = A_Optimized->localMatrix;
	local_int_1d_type matrixDiagonal = A_Optimized->matrixDiagonal;
	int f_numberOfLevels;
	int b_numberOfLevels;
// Forward sweep data and backward sweep data
// map gives us which indixes in _lev_ind are in each level and _lev_ind contains the row numbers.
	local_int_1d_type f_lev_map;
	local_int_1d_type f_lev_ind;
	local_int_1d_type b_lev_map;
	local_int_1d_type b_lev_ind;
// Simple view of length number of rows that holds what level each row is in.
	local_int_1d_type f_row_level;
	local_int_1d_type b_row_level;

	f_numberOfLevels = 0;
	b_numberOfLevels = 0;
// Grab the parts of the matrix A that we need.
	row_map_type matrixRowMap = localMatrix.graph.row_map;
	local_index_type matrixEntries = localMatrix.graph.entries;
	local_int_t nrows = A.localNumberOfRows;
// Allocate the views that need to be allocated right now
	f_row_level = local_int_1d_type("f_row_level", nrows);
	b_row_level = local_int_1d_type("b_row_level", nrows);
// Since this will run in Serial at least for now, create the necessary mirrors for filling row_level
	row_map_type::HostMirror host_matrixRowMap = create_mirror_view(matrixRowMap);
	deep_copy(host_matrixRowMap, matrixRowMap);
	local_index_type::HostMirror host_matrixEntries = create_mirror_view(matrixEntries);
	deep_copy(host_matrixEntries, matrixEntries);
	local_int_1d_type::HostMirror host_f_row_level = create_mirror_view(f_row_level);
	local_int_1d_type::HostMirror host_b_row_level = create_mirror_view(b_row_level);
// Start by taking care of f_row_level
	for(int i = 0; i < nrows; i++){
		int depth = 0;
// Just doing from beginning of row to the diagonal for the forward.
		int start = host_matrixRowMap(i);
		int end = host_matrixRowMap(i+1);
		for(int j = start; j < end; j++){
			int col = host_matrixEntries(j);
			if((col < i) && (host_f_row_level(col) > depth))
				depth = host_f_row_level(col);
		}
		depth++;
		if(depth > f_numberOfLevels) f_numberOfLevels = depth;
		host_f_row_level(i) = depth;
	}
	deep_copy(f_row_level, host_f_row_level); // Copy the host back to the device. I shouldn't need to modify the host anymore so this is fine
// Take care of b_row_level
	for(int i = nrows - 1; i >= 0; i--){
		int depth = 0;
		int start = host_matrixRowMap(i);
		int end = host_matrixRowMap(i+1);
		for(int j = start; j < end; j++){
			int col = host_matrixEntries(j);
			if((col > i) && (host_b_row_level(col) > depth))
				depth = host_b_row_level(col);
		}
		depth++;
		if(depth > b_numberOfLevels) b_numberOfLevels = depth;
		host_b_row_level(i) = depth;
	}
	deep_copy(b_row_level, host_b_row_level);
// Set up f_lev_map and f_lev_ind
	f_lev_map = local_int_1d_type("f_lev_map", f_numberOfLevels+1);
	f_lev_ind = local_int_1d_type("f_lev_ind", nrows);
// Fill up f_lev_map to prepare for scan
	Kokkos::parallel_for(f_numberOfLevels, fillColorsMap(f_lev_map, f_row_level));
// Do the parallel scan on f_lev_map
	Kokkos::parallel_scan(f_numberOfLevels+1, mapScan(f_lev_map));
// Fill our f_lev_ind now.
	Kokkos::parallel_for(f_numberOfLevels, fillColorsInd(f_lev_ind, f_lev_map, f_row_level));
// Set up b_lev_map and b_lev_ind
	b_lev_map = local_int_1d_type("b_lev_map", b_numberOfLevels+1);
	b_lev_ind = local_int_1d_type("b_lev_ind", nrows);
// Fill up b_lev_map to prepare for scan
	Kokkos::parallel_for(b_numberOfLevels, fillColorsMap(b_lev_map, b_row_level));
// Do the parallel scan on f_lev_map
	Kokkos::parallel_scan(b_numberOfLevels+1, mapScan(b_lev_map));
// Fill our b_lev_ind now.
	Kokkos::parallel_for(b_numberOfLevels, fillColorsInd(b_lev_ind, b_lev_map, b_row_level));

	assert(f_lev_map(f_numberOfLevels) == A.localNumberOfRows);

	LevelScheduler * levels = new LevelScheduler;
	levels->f_numberOfLevels = f_numberOfLevels;
	levels->b_numberOfLevels = b_numberOfLevels;
	levels->f_lev_ind = f_lev_ind;
	levels->b_lev_ind = b_lev_ind;
	levels->f_row_level = f_row_level;
	levels->b_row_level = b_row_level;
	levels->f_lev_map = Kokkos::create_mirror_view(f_lev_map);
	Kokkos::deep_copy(levels->f_lev_map, f_lev_map);
	levels->b_lev_map = Kokkos::create_mirror_view(b_lev_map);
	Kokkos::deep_copy(levels->b_lev_map, b_lev_map);
	A_Optimized->levels = levels;
	std::cout<<"F: "<< f_numberOfLevels << " B: "<<b_numberOfLevels << std::endl;
	if(A.Ac != 0) return(levelSchedule(*A.Ac));
	else return(0);

}
#endif