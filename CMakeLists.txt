cmake_minimum_required(VERSION 3.3)
project(AHCI)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++14)
set(SOURCE_FILES main.cpp)
find_package( OpenCV REQUIRED )
add_executable(AHCI ${SOURCE_FILES})
target_link_libraries( AHCI ${OpenCV_LIBS} )
