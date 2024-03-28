################################################################################
# CMake file for pDCX

include_directories(${MPI_CXX_INCLUDE_PATH})
set(LIBS ${MPI_CXX_LIBRARIES})

set(SOURCES mpi_dc.cpp)

add_executable(pDCX ${SOURCES})
target_link_libraries(pDCX ${LIBS})

add_executable(kampingDCX
  kamping_dc.cpp)
target_link_libraries(kampingDCX
  ${LIBS}
  kamping)

################################################################################
