cmake_minimum_required(VERSION 2.8)
# CMAKE_POLICY(SET CMP0048 NEW)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES} )
endif()

set(VERSION_MAJOR   1   CACHE STRING "Project major version number.")
set(VERSION_MINOR   0   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   0   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

project(LR_MNIST)

# it is true that -Ofast activate -ffast-math... but it is to be clear about this point
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=gnu++11 -Wall -ffast-math -fopenmp -Ofast" CACHE STRING " Flags for compile Release" FORCE)
SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -std=gnu++11 -Wall -ffast-math -fno-openmp " CACHE STRING "Flags for compile Debug" FORCE)
#SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -std=gnu++11 -Wall -ffast-math -fopenmp " CACHE STRING "Flags for compile Debug" FORCE)

IF("${PROJECT_BINARY_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
   MESSAGE(FATAL_ERROR "Building in the source tree is not allowed !! Quit; remove the file 'CMakeCache.txt' and the folder 'CMakeFiles' and build outside the sources")
ENDIF("${PROJECT_BINARY_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")


INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(src)
add_subdirectory(docs)
