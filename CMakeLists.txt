cmake_minimum_required(VERSION 3.4.1)

project(test)

add_definitions(-std=c++14 -g)

FILE(GLOB ALL_SOURCES 
    "./src/test.cpp" 
)

FILE(GLOB ALL_INCLUDES 
    "./src/*.h" 
)

set(ALL_SRCS ${ALL_SOURCES} ${ALL_INCLUDES})

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(src/)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(test ${ALL_SRCS})
target_link_libraries(test ${OpenCV_LIBS} ${PCL_LIBRARIES} OpenCL)