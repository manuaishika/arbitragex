cmake_minimum_required(VERSION 3.10)
project(ArbitrageX_OrderBook)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Boost REQUIRED COMPONENTS system)
find_package(GTest REQUIRED)

# Include directories
include_directories(${Boost_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})

# Add source files
set(SOURCES
    src/order_book.cpp
    src/tcp_server.cpp
    src/order_processor.cpp
)

# Add executable
add_executable(order_book ${SOURCES})
target_link_libraries(order_book ${Boost_LIBRARIES})

# Add tests
enable_testing()
add_subdirectory(tests) 