cmake_minimum_required(VERSION 3.20)

project(linkmesh_scorec_helper LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(
  SCOREC 2.2.9 EXACT CONFIG REQUIRED
  COMPONENTS
    mds
    apf
    gmi
    pcu
)

add_executable(
  linkmesh-scorec-helper
  main.cc
)

target_compile_features(
  linkmesh-scorec-helper
  PRIVATE
    cxx_std_17
)

target_link_libraries(
  linkmesh-scorec-helper
  PRIVATE
    MPI::MPI_CXX
    SCOREC::mds
    SCOREC::apf
    SCOREC::gmi
    SCOREC::pcu
)

set_target_properties(
  linkmesh-scorec-helper
  PROPERTIES
    CXX_EXTENSIONS OFF
)

include(GNUInstallDirs)
install(TARGETS linkmesh-scorec-helper RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
