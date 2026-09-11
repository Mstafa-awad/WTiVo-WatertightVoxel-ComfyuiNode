# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.24)
project(WTiVoNative VERSION 1.1.1 LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "WTiVo v1.1 native build currently supports Windows x64 only")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "WTiVo requires a 64-bit compiler/toolchain")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
# OpenVDB historically installs a FindOpenVDB.cmake module, while some package
# managers also expose a config package. Support both forms so the source tree
# is not tied to one vcpkg packaging detail.
find_package(OpenVDB CONFIG QUIET)
if(NOT TARGET OpenVDB::openvdb)
  if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    list(APPEND CMAKE_MODULE_PATH
      "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/openvdb")
  endif()
  find_package(OpenVDB REQUIRED)
endif()
if(NOT TARGET OpenVDB::openvdb)
  message(FATAL_ERROR "OpenVDB was found but OpenVDB::openvdb is unavailable")
endif()

set(WTIVO_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/build" CACHE PATH "WTiVo Python extension output directory")
file(MAKE_DIRECTORY "${WTIVO_OUTPUT_DIR}")

pybind11_add_module(wtivo_core MODULE native/core/wtivo_core.cpp)
target_link_libraries(wtivo_core PRIVATE CGAL::CGAL TBB::tbb Eigen3::Eigen)
if(TARGET TBB::tbbmalloc)
  target_link_libraries(wtivo_core PRIVATE TBB::tbbmalloc)
endif()
target_compile_definitions(wtivo_core PRIVATE
  NOMINMAX WIN32_LEAN_AND_MEAN CGAL_LINKED_WITH_TBB EIGEN_MPL2_ONLY)

pybind11_add_module(wtivo_vdb MODULE native/vdb/wtivo_vdb.cpp)
target_link_libraries(wtivo_vdb PRIVATE OpenVDB::openvdb TBB::tbb)
target_compile_definitions(wtivo_vdb PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

if(MSVC)
  # Keep floating-point behavior conservative. No /fp:fast.
  target_compile_options(wtivo_core PRIVATE /O2 /Ob2 /EHsc /bigobj /fp:precise)
  target_compile_options(wtivo_vdb PRIVATE /O2 /Ob2 /EHsc /bigobj /fp:precise)
else()
  target_compile_options(wtivo_core PRIVATE -O3)
  target_compile_options(wtivo_vdb PRIVATE -O3)
endif()

foreach(t wtivo_core wtivo_vdb)
  set_target_properties(${t} PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${WTIVO_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${WTIVO_OUTPUT_DIR}"
  )
  foreach(cfg RELEASE RELWITHDEBINFO DEBUG MINSIZEREL)
    set_target_properties(${t} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${cfg} "${WTIVO_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_${cfg} "${WTIVO_OUTPUT_DIR}"
    )
  endforeach()
endforeach()

message(STATUS "WTiVo Python: ${Python3_EXECUTABLE}")
message(STATUS "WTiVo native modules: ${WTIVO_OUTPUT_DIR}")
message(STATUS "WTiVo CGAL package: GPL open-source path unless separately commercially licensed")
