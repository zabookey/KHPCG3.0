#ifdef SYMGS_COLOR
#include "Coloring.hpp"

class Coloring{
	public:
	typedef local_int_t Ordinal;
	typedef execution_space ExecSpace;
  typedef typename Kokkos::View<Ordinal *, ExecSpace> array_type;
  typedef typename Kokkos::View<Ordinal, ExecSpace> ordinal_type;

  Ordinal _size;
  array_type _idx;
  array_type _adj;
  array_type _colors;
  conflict_type  _conflictType; // Choose at run-time
  array_type _vertexList;
  array_type _recolorList;
  ordinal_type _vertexListLength;  // 0-dim Kokkos::View, so really Ordinal
  ordinal_type _recolorListLength; // 0-dim Kokkos::View, so really Ordinal	

  ordinal_type::HostMirror host_vertexListLength;
  ordinal_type::HostMirror host_recolorListLength;

	Coloring(Ordinal nvtx, array_type idx, array_type adj, array_type colors):
		_size(nvtx), _idx(idx), _adj(adj), _colors(colors){
		// vertexList contains all initial vertices
		_vertexList = array_type("vertexList", nvtx);
		_vertexListLength = ordinal_type("vertexListLength");
		host_vertexListLength = Kokkos::create_mirror_view(_vertexListLength);
		host_vertexListLength = nvtx;
		Kokkos::deep_copy(_vertexListLength, host_vertexListLength);
		// Initialize _vertexList (natural order)
		functorInitList<Ordinal, ExecSpace> init(_vertexList);
		Kokkos::parallel_for(nvtx, init);
		// Vertices to recolor. Will swap with vertexList
		_recolorList = array_type("recolorList", nvtx);
		_recolorListLength = ordinal_type("recolorListLength");
		host_recolorListLength = Kokkos::create_mirror_view(_recolorListLength);
		host_recolorListLength() = 0;
		Kokkos::deep_copy(_recolorListLength, host_recolorListLength);
	}

	void color(bool useConflictList, bool serialConflictResolution, bool ticToc){
		Ordinal numUncolored = _size; // on host
		double t, total = 0.0;
		Kokkos::Impl::Timer timer;

		if(useConflictList)
			_conflictType = CONFLICT_LIST;

		// While vertices to color, do speculative coloring.
		int iter = 0;
		for(iter = 0; (iter<20) && (numUncolored>0); iter++){
			std::cout<< "Start iteration " << iter << std::endl;

			// First color greedy speculatively, some conflicts expected
			this -> colorGreedy();
			ExecSpace::fence();
			if(ticToc){
				t = timer.seconds();
				total += t;
				std::cout << "Time speculative greedy phase " << iter << " : " << std::endl;
				timer.reset();
			}

#ifdef DEBUG
			// UVM required - will be slow!
			printf("\n 100 first vertices: ");
			for(int i = 0; i < 100; i++){
				printf(" %i", _colors[i]);
			}
			printf("\n");
#endif

			// Check for conflicts (parallel), find vertices to recolor
			numUncolored = this -> findConflicts();

			ExecSpace::fence();
			if(ticToc){
			t = timer.seconds();
			total += t;
			std::cout << "Time conflict detection " << iter << " : " << t << std::endl;
			timer.reset();
			}
			if (serialConflictResolution) break; // Break after first iteration
/*			if(_conflictType == CONFLICT_LIST){
				array_type temp = _vertexList;
				_vertexList = _recolorList;
				_vertexListLength() = _recolorListLength();
				_recolorList = temp;
				_recolorListLength() = 0;
			}
*/			if(_conflictType == CONFLICT_LIST){
				array_type temp = _vertexList;
				_vertexList = _recolorList;
				host_vertexListLength() = host_recolorListLength();
				_recolorList = temp;
				host_recolorListLength() = 0;
				Kokkos::deep_copy(_vertexListLength, host_vertexListLength);
				Kokkos::deep_copy(_recolorListLength, host_recolorListLength);
			}
		}

		std::cout << "Number of coloring iterations: " << iter << std::endl;

		if(numUncolored > 0){
			// Resolve conflicts by recolor in serial
			this -> resolveConflicts();
			ExecSpace::fence();
			if(ticToc){
				t = timer.seconds();
				total += t;
				std::cout << "Time conflict resolution: " << t << std::endl;
				std::cout << "Total time: " << total << std::endl;
			}
		}
	}

	void colorGreedy(){
		Ordinal chunkSize = 8; // Process chunkSize vertices in one chunk
		if(host_vertexListLength < 100*chunkSize)
			chunkSize = 1;

		functorGreedyColor<Ordinal, ExecSpace> gc(_idx, _adj, _colors, _vertexList, host_vertexListLength, chunkSize);
		Kokkos::parallel_for(host_vertexListLength/chunkSize+1, gc);
		Kokkos::deep_copy(host_vertexListLength, _vertexListLength);
	}

	Ordinal findConflicts(){
		functorFindConflicts<Ordinal, ExecSpace> conf(_idx, _adj, _colors, _vertexList, _recolorList, _recolorListLength, _conflictType);
		Ordinal numUncolored;
		Kokkos::parallel_reduce(host_vertexListLength(), conf, numUncolored);
		Kokkos::deep_copy(host_recolorListLength, _recolorListLength);
		std::cout<< "Number of uncolored vertices: " << numUncolored << std::endl;
#ifdef DEBUG
		if(_conflictType == CONFLICT_LIST)
			std::cout << "findConflicts: recolorListLength = " << host_recolorListLength() << std::endl;
#endif
		return numUncolored;
	}

	void resolveConflicts(){
	// This method is in serial so it will need a bit of reworking to be used on Cuda.
		// Compute maxColor.
		const int maxColor = 255; // Guess, since too expensive to loop over nvtx
		
		int forbidden[maxColor+1];
		Ordinal i = 0;
		for(Ordinal k = 0; k < _size; k++){
			if(_conflictType == CONFLICT_LIST){
				if(k == host_recolorListLength()) break;
				i = _recolorList[k];
			}
			else {
				// Check for uncolored vertices
				i = k;
				if (_colors[i] > 0) continue;
			}

			// recolor vertex i with smallest available color
			
			// check neighbors
			for(Ordinal j = _idx[i]; j < _idx[i+1]; j++){
				if(_adj[j] == i) continue; // Skip self-loops
				forbidden[_colors[_adj[j]]] = i;
			}
			// color vertex i with smallest available color
			int c=1;
			while((forbidden[c] == i) && c<= maxColor) c++;
			_colors[i] = c;
		}
	}

	Ordinal getNumColors() {
      Ordinal maxColor=0;
      // TODO: parallel_reduce? This produced strange results... So instead we'll use mirrors and only call this method once and store the result
			array_type::HostMirror _colors_host = Kokkos::create_mirror_view(_colors);
			Kokkos::deep_copy(_colors_host, _colors);
			for(int i = 0; i < _size; i++)
				if(_colors_host(i) > maxColor) maxColor = _colors_host(i);
			return maxColor;
    }
};

class fillIdx{
	public:
	Coloring::array_type idx;
	row_map_type row_map;

	fillIdx(Coloring::array_type& idx_, row_map_type& row_map_):
		idx(idx_), row_map(row_map_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		idx(i) = row_map(i);
	}
};

class fillAdj{
	public:
	Coloring::array_type adj;
	local_index_type indices;

	fillAdj(Coloring::array_type& adj_, local_index_type& indices_):
		adj(adj_), indices(indices_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		adj(i) = indices(i);
	}
};

class fillColorsMap{
	public:
	local_int_1d_type colors_map;
	Coloring::array_type colors;

	fillColorsMap(const local_int_1d_type& colors_map_, const Coloring::array_type& colors_):
		colors_map(colors_map_), colors(colors_){}

	// This fills colors_map(i) with the number of indices with color i. Parallel scan will make this the appropriate map.
	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		int color = i+1; // Since i starts at 0 and colors start at 1.
		int total = 0;
		for(int j = 0; j < colors.dimension_0(); j++)
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
	Coloring::array_type colors;

	fillColorsInd(local_int_1d_type& colors_ind_, local_int_1d_type& colors_map_,
		Coloring::array_type colors_):
		colors_ind(colors_ind_), colors_map(colors_map_), colors(colors_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		int color = i+1; // Colors start at 1 and i starts at 0.
		int start = colors_map(i);
		for(int j = 0; j < colors.dimension_0(); j++){
			if(colors(j) == color){
				colors_ind(start) = j;
				start++;
			}
		}
		assert(start == colors_map(i+1));//Make sure we only fill up exactly our color. Nothing more nothing less.
	}
};

int doColoring(SparseMatrix & A){
	assert(A.optimizationData != 0);
	Optimatrix * A_Optimized = (Optimatrix*) A.optimizationData;
	local_matrix_type localMatrix = A_Optimized->localMatrix;
	Coloring::array_type colors("colors", A.localNumberOfRows);
	Coloring::array_type idx("idx", localMatrix.graph.row_map.dimension_0()); // Should be A.localNumberOfRows+1 length
	Coloring::array_type adj("adj", localMatrix.graph.entries.dimension_0()); // Should be A.LocalNumberOfNonzeros.
	Kokkos::parallel_for(localMatrix.graph.row_map.dimension_0(), fillIdx(idx, localMatrix.graph.row_map));
	Kokkos::parallel_for(localMatrix.graph.entries.dimension_0(), fillAdj(adj, localMatrix.graph.entries));
	Coloring c(A.localNumberOfRows, idx, adj, colors);
	c.color(false, false, false); // Flags are as follows... Use conflict List, Serial Resolve Conflict, Time and show.
	int numColors = c.getNumColors();
	local_int_1d_type colors_map("Colors Map", numColors + 1);
	local_int_1d_type colors_ind("Colors Idx", A.localNumberOfRows);
// Fill colors_map so that colors_map(i) contains the number of entries with color i
	Kokkos::parallel_for(numColors, fillColorsMap(colors_map, colors));
// Scan colors_map to finish filling out the map.
	Kokkos::parallel_scan(numColors + 1, mapScan(colors_map));
// Use colors_map to fill fill out colors_ind.
	Kokkos::parallel_for(numColors, fillColorsInd(colors_ind, colors_map, colors));

// Assign everything back to A now.
	A_Optimized->colors_map = colors_map;
	Coloring::array_type::HostMirror host_colors_map = Kokkos::create_mirror_view(colors_map);
	Kokkos::deep_copy(host_colors_map, colors_map);
	A_Optimized->host_colors_map = host_colors_map;
	A_Optimized->colors_ind = colors_ind;
	Coloring::array_type::HostMirror host_colors_ind = Kokkos::create_mirror_view(colors_ind);
	Kokkos::deep_copy(host_colors_ind, colors_ind);
	A_Optimized->host_colors_ind = host_colors_ind;
	A_Optimized->numColors = numColors;
	if(A.Ac != 0) return doColoring(*A.Ac);
	return(0);
}
#endif