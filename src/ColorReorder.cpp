#ifdef REORDER
#include "KokkosSetup.hpp"
#include "Vector.hpp"
#include "SparseMatrix.hpp"

#include <iostream>

class PermuteRow{
public:
	local_matrix_type localMatrix;
	values_type newValues;
	local_index_type newIndices;
	int old_row_start;
	int new_row_start;

	PermuteRow(const local_matrix_type & localMatrix_, values_type & newValues_, local_index_type & newIndices_,
		const int old_row_start_, const int new_row_start_):
		localMatrix(localMatrix_), newValues(newValues_), newIndices(newIndices_),
		old_row_start(old_row_start_), new_row_start(new_row_start_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i) const{
		newValues(new_row_start + i) = localMatrix.values(old_row_start + i);
		newIndices(new_row_start + i) = localMatrix.graph.entries(old_row_start + i);
	}
};

class PermuteColumn{
public:
	local_int_1d_type row_dest;
	local_index_type newIndices;

	PermuteColumn(const local_int_1d_type & row_dest_, local_index_type &newIndices_):
		row_dest(row_dest_), newIndices(newIndices_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i) const{
		newIndices(i) = row_dest(newIndices(i));
	}
};

class UpdateDiagonal{
public:
	local_int_1d_type new_diag;
	row_map_type newRowMap;
	local_index_type newIndices;

	UpdateDiagonal(local_int_1d_type & new_diag_, const row_map_type & newRowMap_, const local_index_type & newIndices_):
		new_diag(new_diag_), newRowMap(newRowMap_), newIndices(newIndices_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i) const{
		for(int j = newRowMap(i); j < newRowMap(i+1); j++){
			if(newIndices(j) == i){
				new_diag(i) = j;
				break;
			}
		}
	}

};

/*
 * This function reorders our problem based on the colors the matrix is given
 * This should increase performance in ColorSYMGS by increasing cache locality?
 * FIXME This function may not update everything it needs to yet and needs testing
 */

int ColorReorder(SparseMatrix & A, Vector & x, Vector & b){
	Optimatrix* A_Optimized = (Optimatrix *) A.optimizationData;
	local_matrix_type localMatrix = A_Optimized->localMatrix;
	local_int_1d_type matrixDiagonal = A_Optimized->matrixDiagonal;
	int numColors = A_Optimized->numColors;
	host_local_int_1d_type host_colors_map = A_Optimized->host_colors_map;
	host_local_int_1d_type host_colors_ind = A_Optimized->host_colors_ind;
	Optivector * x_Optimized = (Optivector *) x.optimizationData;
	double_1d_type x_values = x_Optimized->values;
	Optivector * b_Optimized = (Optivector *) b.optimizationData;
	double_1d_type b_values = b_Optimized->values;
	//WHOO!!!! BABY STEPS!
	std::cout<<"REORDERING..." << std::endl;
	values_type newValues("New Values on Device", localMatrix.values.dimension_0());
	local_index_type newIndices("New entries on Device", localMatrix.graph.entries.dimension_0());
	non_const_row_map_type newRowMap("New RowMap on Device", localMatrix.graph.row_map.dimension_0());
	host_non_const_row_map_type host_newRowMap = Kokkos::create_mirror_view(newRowMap); // ASSUME THIS SETS IT TO ALL 0's
	double_1d_type newX("New X_values on Device", x_values.dimension_0());
	host_double_1d_type host_newX = Kokkos::create_mirror_view(newX);
	double_1d_type newB("New B values on Device", b_values.dimension_0());
	host_double_1d_type host_newB = Kokkos::create_mirror_view(newB);
	local_int_1d_type orig_rows = local_int_1d_type("Original row indices", A.localNumberOfRows);
	host_local_int_1d_type host_orig_rows = Kokkos::create_mirror_view(orig_rows);
	local_int_1d_type row_dest = local_int_1d_type("Row Destinatinos", A.localNumberOfRows); //row_dest(i) = old row i's new location
	host_local_int_1d_type host_row_dest = Kokkos::create_mirror_view(row_dest);
	local_int_t destinationRow = 0;
	// Outer loop iterates through the colors
	for(int i = 0; i < numColors; i++){
		local_int_t begin = host_colors_map(i);
		local_int_t end = host_colors_map(i+1);
		// Middle Loop iterates through the rows that belong to each color
		for(local_int_t j = begin; j < end; j++){
			//Move Matrix Row
			local_int_t currentRow = host_colors_ind(j);
			local_int_t old_row_start = localMatrix.graph.row_map(currentRow);
			local_int_t old_row_end = localMatrix.graph.row_map(currentRow+1);
			local_int_t nnzInRow = old_row_end - old_row_start; //Mark how many nonzeros are in the row we're about to move
			local_int_t new_row_start = host_newRowMap(destinationRow);
			// Innermost loop iterates through the values in the row
			//TODO This loop can and should be done in parallel
			/*
			for(local_int_t k = 0; k < old_row_end - old_row_start; k++){
				newValues(new_row_start + k) = localMatrix.values(k + old_row_start);
				newIndices(new_row_start + k) = localMatrix.graph.entries(k + old_row_start);
			}
			*/
			Kokkos::parallel_for(old_row_end - old_row_start, PermuteRow(localMatrix, newValues, newIndices, old_row_start, new_row_start));
			//RHS may need to be mirrors instead of actual view
			host_newX(destinationRow) = x_values(currentRow);
			host_newB(destinationRow) = b_values(currentRow);
			host_orig_rows(destinationRow) = currentRow; // Mark which row of this was the original so we can return it later.
			host_row_dest(currentRow) = destinationRow;
			//Prepare newRowMap for the next iteration
			host_newRowMap(destinationRow+1) = host_newRowMap(destinationRow) + nnzInRow;
			//Increment destination so we continue to fill our new mappings
			destinationRow++;
		}
	}
	Kokkos::deep_copy(newRowMap, host_newRowMap);
	Kokkos::deep_copy(row_dest, host_row_dest);
	Kokkos::deep_copy(orig_rows, host_orig_rows);
	Kokkos::deep_copy(newX, host_newX);
	Kokkos::deep_copy(newB, host_newB);
	//This will permute the columns to help us maintain symmetry.
	/*
	for(int i = 0; i < newIndices.dimension_0(); i++)
		newIndices(i) = host_row_dest(newIndices(i));
	*/
	Kokkos::parallel_for(newIndices.dimension_0(), PermuteColumn(row_dest, newIndices));
	local_int_1d_type new_diag("New Diagonal", A.localNumberOfRows);
	/*
	for(int i = 0; i < A.localNumberOfRows; i++){
		for(int j = newRowMap(i); j < newRowMap(i+1); j++){
			if(newIndices(j) == i){
				new_diag(i) = j;
				break;
			}
		}
	}
	*/
	Kokkos::parallel_for(A.localNumberOfRows, UpdateDiagonal(new_diag, newRowMap, newIndices));
	A_Optimized->matrixDiagonal = new_diag;
	A_Optimized->orig_rows = orig_rows;
	A_Optimized->row_dest = row_dest;
	local_matrix_type reordered_matrix = local_matrix_type("Matrix: Reordered", A.localNumberOfRows, A.localNumberOfRows, A.localNumberOfNonzeros,
		newValues, newRowMap, newIndices);
	A_Optimized->localMatrix = reordered_matrix;
	x_Optimized->values = newX;
	b_Optimized->values = newB;
	//Kokkos::deep_copy(b_values, newB);
	//Kokkos::deep_copy(x_values, newX);
	//Kokkos::deep_copy(A.globalMatrix.values, newValues);
	//Kokkos::deep_copy(A.globalMatrix.graph.entries, newIndices);
	//Kokkos::deep_copy(A.globalMatrix.graph.row_map, newRowMap);
}
#endif
