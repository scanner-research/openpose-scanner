# To build your custom op, you can either use our CMake convenience methods
# or do it the slightly harder way in normal Make (see the Makefile).

cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

execute_process(
  OUTPUT_VARIABLE SCANNER_CMAKE_DIR
  COMMAND python -c "import scannerpy.stdlib.build_flags as b; b.get_cmake()")
include(${SCANNER_CMAKE_DIR}/Op.cmake)

# build_op will create a shared library called lib${LIB_NAME}.so that builds
# from all of the CPP_SRCS. You can also optionally specify a PROTO_SRC that
# points to a Protobuf file and will generate the C++ and Python bindings.
build_op(
  LIB_NAME openpose_op
  CPP_SRCS openpose_op.cpp
  PROTO_SRC openpose.proto)

# The library specified in build_op is a normal CMake target, so you can use all
# the normal CMake functions with it.
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
target_link_libraries(openpose_op PUBLIC "${OpenCV_LIBRARIES}")

include_directories(SYSTEM /opt/openpose/include)
target_link_libraries(openpose_op PUBLIC /opt/openpose/lib/libopenpose.so)
