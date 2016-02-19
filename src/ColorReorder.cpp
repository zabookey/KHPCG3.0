#ifdef REORDER
#include "KokkosSetup.hpp"
#include "Vector.hpp"
#include "SparseMatrix.hpp"

#include <iostream>

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
	host_values_type newValues = Kokkos::create_mirror(localMatrix.values);
	host_local_index_type newIndices = Kokkos::create_mirror(localMatrix.graph.entries);
	host_non_const_row_map_type newRowMap = Kokkos::create_mirror(localMatrix.graph.row_map); // ASSUME THIS SETS IT TO ALL 0's
	host_double_1d_type newX = Kokkos::create_mirror(x_values);
	host_double_1d_type newB = Kokkos::create_mirror(b_values);
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
			local_int_t new_row_start = newRowMap(destinationRow);
			// Innermost loop iterates through the values in the row
			//TODO This loop can and should be done in parallel
			for(local_int_t k = 0; k < old_row_end - old_row_start; k++){
				newValues(new_row_start + k) = localMatrix.values(k + old_row_start);
				newIndices(new_row_start + k) = localMatrix.graph.entries(k + old_row_start);
			}
			//RHS may need to be mirrors instead of actual view
			newX(destinationRow) = x_values(currentRow);
			newB(destinationRow) = b_values(currentRow);
			host_orig_rows(destinationRow) = currentRow; // Mark which row of this was the original so we can return it later.
			host_row_dest(currentRow) = destinationRow;
			//Prepare newRowMap for the next iteration
			newRowMap(destinationRow+1) = newRowMap(destinationRow) + nnzInRow;
			//Increment destination so we continue to fill our new mappings
			destinationRow++;
		}
	}
	//This will permute the columns to help us maintain symmetry.
	for(int i = 0; i < newIndices.dimension_0(); i++)
		newIndices(i) = host_row_dest(newIndices(i));
	local_int_1d_type new_diag = local_int_1d_type("New Diagonal", A.localNumberOfRows);
	for(int i = 0; i < A.localNumberOfRows; i++){
		for(int j = newRowMap(i); j < newRowMap(i+1); j++){
			if(newIndices(j) == i){
				new_diag(i) = j;
				break;
			}
		}
		if(i == A.localNumberOfRows -1) std::cout<<"Diagonal redone"<<std::endl;
	}
	A_Optimized->matrixDiagonal = new_diag;
	double_1d_type reordered_x_values = double_1d_type("Reordered x", newX.dimension_0());
	Kokkos::deep_copy(reordered_x_values, newX);
	double_1d_type reordered_b_values = double_1d_type("Reordered b", newB.dimension_0());
	Kokkos::deep_copy(reordered_b_values, newB);
	x_values = reordered_x_values;
	//b_values = reordered_b_values;
	Kokkos::deep_copy(orig_rows, host_orig_rows);
	A_Optimized->orig_rows = orig_rows;
	values_type newV = values_type("New Values on Device", newValues.dimension_0());
	local_index_type newI = local_index_type("New Indices on Device", newIndices.dimension_0());
	non_const_row_map_type newR = non_const_row_map_type("New Row Map on device", newRowMap.dimension_0());
	Kokkos::deep_copy(newV, newValues);
	Kokkos::deep_copy(newI, newIndices);
	Kokkos::deep_copy(newR, newRowMap);
	local_matrix_type reordered_matrix = local_matrix_type("Matrix: Reordered", A.localNumberOfRows, A.localNumberOfRows, A.localNumberOfNonzeros, newV, newR, newI);
	A_Optimized->localMatrix = reordered_matrix;
	x_Optimized->values = reordered_x_values;
	b_Optimized->values = reordered_b_values;
	//Kokkos::deep_copy(b_values, newB);
	//Kokkos::deep_copy(x_values, newX);
	//Kokkos::deep_copy(A.globalMatrix.values, newValues);
	//Kokkos::deep_copy(A.globalMatrix.graph.entries, newIndices);
	//Kokkos::deep_copy(A.globalMatrix.graph.row_map, newRowMap);
}
#endif
