# pgr4_recomp - ReXGlue Recompiled Project
#
# SDK boilerplate lives in generated/rexglue.cmake.

cmake_minimum_required(VERSION 3.25)
project(pgr4_recomp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(generated/rexglue.cmake)

# Sources
set(PGR4RECOMP_SOURCES
    src/main.cpp
)

if(WIN32)
    add_executable(pgr4_recomp WIN32 ${PGR4RECOMP_SOURCES})
else()
    add_executable(pgr4_recomp ${PGR4RECOMP_SOURCES})
endif()

rexglue_setup_target(pgr4_recomp)
