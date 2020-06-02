cmake_minimum_required(VERSION 3.8..3.15) # CMake will act as version 3.15 and if unavailable 3.8
project(mcexercise CXX) # Name of Python module, must match with the name in the cpp file

find_package(Python3 COMPONENTS Development REQUIRED) # Find the Python libs

# Abbreviate the name to whatever Python version was found on the system
set(BOOST_PYTHON_COMPONENT python3${Python3_VERSION_MINOR})
find_package(Boost COMPONENTS ${BOOST_PYTHON_COMPONENT} numpy REQUIRED) # Find Boost.Python and Boost.NumPy

Python3_add_library  (mcexercise MODULE main.cpp) # We are compiling a Python module
# This module needs Boost.Python, Boost.NumPy and Python itself
target_link_libraries(mcexercise PRIVATE Boost::${BOOST_PYTHON_COMPONENT} Boost::numpy Python3::Python)
# After the so is built place it in the same folder as CMakeLists.txt
set_target_properties(mcexercise PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

target_compile_features(mcexercise PUBLIC cxx_std_20) # Use C++2a
# Enable many warnings, and don't finish compiling when something's wrong
target_compile_options (mcexercise PRIVATE -Wall -Wextra -Wpedantic -Werror) # TODO -Weverything -Wno-c++98-compat with clang
target_compile_options (mcexercise PRIVATE -O3 -march=native -flto -fconcepts) # Enable optimizations
target_link_options    (mcexercise PRIVATE -flto -fuse-ld=gold)     # Enable optimizations and 'gold' linker, for clang

# TODO Enable static analyzers, sanitizers, formatting, more warnings, debug/release specific features, ...
