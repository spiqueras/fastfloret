cmake_minimum_required(VERSION 3.18...3.31)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.11
  REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(FLORET_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
file(GLOB FLORET_SOURCES CONFIGURE_DEPENDS
  "${FLORET_SRC_DIR}/*.cc" "${FLORET_SRC_DIR}/*.cpp")
list(REMOVE_ITEM FLORET_SOURCES "${FLORET_SRC_DIR}/main.cc")

# The core library is built separately so it gets full optimisation;
# nanobind_add_module optimises its sources for size.
add_library(floret_core STATIC ${FLORET_SOURCES})
set_target_properties(floret_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(floret_core PUBLIC "${FLORET_SRC_DIR}")
target_link_libraries(floret_core PUBLIC Threads::Threads)
if(NOT MSVC)
  target_compile_options(floret_core PRIVATE -O3 -funroll-loops)
endif()

nanobind_add_module(fastfloret_pybind
  STABLE_ABI
  FREE_THREADED
  fastfloret_module/fastfloret/pybind/fastfloret_pybind.cc)
target_link_libraries(fastfloret_pybind PRIVATE floret_core)

install(TARGETS fastfloret_pybind LIBRARY DESTINATION .)
