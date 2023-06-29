cmake_minimum_required (VERSION 3.16)

project(onnx-benchmark LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(onnxruntime REQUIRED)
find_package(CLI11 REQUIRED)

add_executable(onnx-cpp-benchmark onnx-cpp-benchmark.cpp)

target_link_libraries(onnx-cpp-benchmark PRIVATE onnxruntime::onnxruntime CLI11::CLI11)
