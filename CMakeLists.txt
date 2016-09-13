project(multiTracking)
cmake_minimum_required(VERSION 2.8)
option(BUILD_TRAX_CLIENT "Build the trax client." OFF)
find_package(OpenCV REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(multiTracking multitracking.cpp CMT.h CMT.cpp)

target_link_libraries(multiTracking ${OpenCV_LIBS})
