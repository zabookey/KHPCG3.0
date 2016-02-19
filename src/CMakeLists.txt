set(SOURCES
  #Put all of the .cpp files here except for main
  CG.cpp  CG_ref.cpp  CheckAspectRatio.cpp  CheckProblem.cpp  ComputeDotProduct.cpp  ComputeDotProduct_ref.cpp  ComputeMG.cpp  ComputeMG_ref.cpp
  ComputeOptimalShapeXYZ.cpp  ComputeProlongation_ref.cpp  ComputeResidual.cpp  ComputeRestriction_ref.cpp ComputeSPMV.cpp
  ComputeSPMV_ref.cpp ComputeSYMGS.cpp ComputeSYMGS_ref.cpp ComputeWAXPBY.cpp ComputeWAXPBY_ref.cpp
  ExchangeHalo.cpp  finalize.cpp  GenerateCoarseProblem.cpp  GenerateGeometry.cpp  GenerateProblem.cpp  GenerateProblem_ref.cpp
  init.cpp  MixedBaseCounter.cpp mytimer.cpp  OptimizeProblem.cpp  ReadHpcgDat.cpp  ReportResults.cpp  SetupHalo.cpp  SetupHalo_ref.cpp
  TestCG.cpp  TestNorms.cpp  TestSymmetry.cpp  WriteProblem.cpp  YAML_Doc.cpp
  YAML_Element.cpp  Coloring.cpp ColorSYMGS.cpp ColorReorder.cpp LevelScheduler.cpp LevelSYMGS.cpp
  )

set(HEADERS
  #Put all of the .hpp files here
  CGData.hpp  CG.hpp  CG_ref.hpp  CheckAspectRatio.hpp  CheckProblem.hpp  ComputeDotProduct.hpp  ComputeDotProduct_ref.hpp  ComputeMG.hpp
  ComputeMG_ref.hpp  ComputeOptimalShapeXYZ.hpp ComputeProlongation_ref.hpp  ComputeResidual.hpp  ComputeRestriction_ref.hpp
  ComputeSPMV.hpp ComputeSPMV_ref.hpp  ComputeSYMGS.hpp  ComputeSYMGS_ref.hpp  ComputeWAXPBY.hpp
  ComputeWAXPBY_ref.hpp  ExchangeHalo.hpp  GenerateCoarseProblem.hpp GenerateGeometry.hpp  
  GenerateProblem.hpp  GenerateProblem_ref.hpp Geometry.hpp  hpcg.hpp KokkosSetup.hpp  MGData.hpp  MixedBaseCounter.hpp mytimer.hpp
  OptimizeProblem.hpp  ReadHpcgDat.hpp  ReportResults.hpp  SetupHalo.hpp  SetupHalo_ref.hpp SparseMatrix.hpp  TestCG.hpp
  TestNorms.hpp  TestSymmetry.hpp  Vector.hpp  WriteProblem.hpp  YAML_Doc.hpp  YAML_Element.hpp
  Coloring.hpp ColorSYMGS.hpp ColorReorder.cpp Levels.hpp LevelScheduler.hpp LevelSYMGS.hpp
	
  )

message("Compiler Flags: ${CMAKE_CXX_FLAGS}")

include_directories(./ ./src_subdirTBD
                    ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})

link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

add_library(kokkoshpcglib ${SOURCES} ${HEADERS})

add_executable(KokkosHPCG.exe main.cpp)

file(COPY ../hpcg.dat DESTINATION .)

target_link_libraries(KokkosHPCG.exe kokkoshpcglib ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
